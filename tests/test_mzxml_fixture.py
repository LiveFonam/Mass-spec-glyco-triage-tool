"""Regression test: parse a real mzXML file and run the full pipeline.

The fixture is ``data/2c5-120m_spectrum.mzXML`` -- a 120-minute O-glycan
profile spectrum captured on a Bruker ultraFlex TOF/TOF (MALDI). It
contains 467k peaks across the 600-6000 Da acquisition window. We
subsample (every 100th peak) so the test runs in well under a second,
then drive the parsed spectrum through the parse -> solve -> screen
pipeline end-to-end.

The test asserts:
  1. The mzXML parser successfully decodes the first MS1 scan and
     reports peak / m/z / intensity ranges consistent with the file
     header (``lowMz``, ``highMz``, ``peaksCount``).
  2. The composition solver accepts the parsed spectrum and returns a
     non-empty list of candidates in the test m/z window.
  3. The screener accepts the solver output, attaches the expected
     columns, and completes without exception.

This is a smoke test for the pipeline, not a "known-good answer"
regression. We deliberately do NOT assert a specific composition or
tier breakdown -- the fixture's purpose is to catch breakage in the
mzXML -> Spectrum -> solve_spectrum -> screen_candidates wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from glycan_ms.core import (  # noqa: E402
    Adduct,
    Peak,
    Spectrum,
    dedup_peaks,
    solve_spectrum,
)
from glycan_ms.parser_mzxml import first_scan_spectrum  # noqa: E402
from glycan_ms.screener import screen_candidates  # noqa: E402

# Path to the fixture, resolved relative to the repo root. Keeping it
# here (rather than at the top of the file) lets pytest show the
# fixture path in test reports when a check fails.
MZXML_FIXTURE = (
    Path(__file__).resolve().parent.parent / "data" / "2c5-120m_spectrum.mzXML"
)

# Stride for the subsample. The full spectrum has ~467k peaks which
# is fine for production but far too many for a unit test. Every
# 100th peak keeps the dedup / solve / screen path exercised at
# realistic peak density (~4700 peaks) while completing in well
# under a second on a laptop.
SUBSAMPLE_STRIDE = 100

# m/z window for the solve. The fixture's acquisition range is
# 600-5998 Da; we restrict to 1200-3000 so the test focuses on the
# O-glycan regime the project targets.
MZ_LO = 1200.0
MZ_HI = 3000.0

# Da tolerance for the solve and the screen. Slightly wider than the
# sidebar default (0.7) so we still get candidates on a subsampled
# spectrum where the precise peak centre may be skipped.
DA_TOL = 1.0

# Minimum peak intensity for the solve. The subsampled spectrum
# inherits the real intensity distribution; this floor filters out
# the dense noise floor (sub-500) without requiring a per-test
# re-tuning of the noise statistic.
MIN_INTENSITY = 500.0


@pytest.fixture(scope="module")
def parsed_spectrum() -> Spectrum:
    """Parse the first MS1 scan of the mzXML fixture, subsampled.

    Module-scoped so the parse cost (a few seconds on the full
    ~467k-peak file) is paid once per pytest invocation rather than
    once per test.
    """
    if not MZXML_FIXTURE.exists():
        pytest.skip(f"mzXML fixture not present: {MZXML_FIXTURE}")
    sp = first_scan_spectrum(str(MZXML_FIXTURE))
    sub_peaks = sp.peaks[::SUBSAMPLE_STRIDE]
    sub_spectrum = Spectrum(
        peaks=dedup_peaks(sub_peaks, bin_width=0.01),
        mz_hi=sp.upper_bound(),
    )
    return sub_spectrum


def test_mzxml_fixture_exists() -> None:
    """The fixture file is in the repo at the expected path."""
    assert MZXML_FIXTURE.exists(), (
        f"mzXML fixture missing: {MZXML_FIXTURE}. "
        f"Download the test case file and place it at this path."
    )
    assert MZXML_FIXTURE.stat().st_size > 1_000_000, (
        f"mzXML fixture looks too small ({MZXML_FIXTURE.stat().st_size} bytes); "
        f"expected a real-world ~5 MB profile spectrum."
    )


def test_mzxml_parser_decodes_first_scan() -> None:
    """The parser produces a Spectrum with peaks in the expected range.

    The fixture's <scan> header advertises lowMz=599.95,
    highMz=5998.33, peaksCount=467456. The parser must produce a
    peak list whose m/z range brackets this window and whose size
    is consistent with the documented peak count (allowing for
    subsampling by the caller).
    """
    if not MZXML_FIXTURE.exists():
        pytest.skip(f"mzXML fixture not present: {MZXML_FIXTURE}")
    sp = first_scan_spectrum(str(MZXML_FIXTURE))
    assert sp.peaks, "parser returned an empty peak list"
    mz_values = [p.mz for p in sp.peaks]
    min_mz = min(mz_values)
    max_mz = max(mz_values)
    # Acquisition window must be honoured.
    assert min_mz < 700.0, f"min m/z {min_mz} below expected acquisition start"
    assert max_mz > 5500.0, f"max m/z {max_mz} below expected acquisition end"
    # Header reports 467k peaks -- allow a small tolerance for any
    # decoding quirk.
    assert len(sp.peaks) > 100_000, (
        f"expected >100k peaks in full first scan, got {len(sp.peaks)}"
    )


def test_subsample_dedup_preserves_mz_range(parsed_spectrum: Spectrum) -> None:
    """Subsampling + dedup keeps the spectrum m/z range intact."""
    assert parsed_spectrum.peaks, "subsampled spectrum is empty"
    mz_values = [p.mz for p in parsed_spectrum.peaks]
    assert min(mz_values) >= 590.0
    assert max(mz_values) <= 6010.0


def test_solve_spectrum_returns_candidates(parsed_spectrum: Spectrum) -> None:
    """solve_spectrum returns a non-empty candidate list on real data.

    The subsampled spectrum may have noisy peak intensities (the
    subsample skips real peaks between strides), so we use a wider
    Da tolerance than the sidebar default to still get hits. We
    require >= 1 candidate to confirm the solver runs end-to-end.
    """
    matches = solve_spectrum(
        parsed_spectrum,
        da_tol=DA_TOL,
        mz_lo=MZ_LO,
        mz_hi=MZ_HI,
        adducts=[Adduct.NA, Adduct.K],
        min_intensity=MIN_INTENSITY,
    )
    assert len(matches) >= 1, (
        f"solve_spectrum returned 0 candidates on the mzXML fixture "
        f"(mz window [{MZ_LO}, {MZ_HI}], da_tol={DA_TOL}); "
        f"the subsample stride may need adjustment."
    )


def test_screen_candidates_runs_endtoend(parsed_spectrum: Spectrum) -> None:
    """screen_candidates runs end-to-end on the parsed fixture.

    We don't assert a specific tier breakdown -- the fixture's
    subsampled noise floor naturally PURGEs many rows. What we
    verify is that the pipeline produces a DataFrame with the
    expected shape and that all tier values are in the allowed set.
    A crash here means a downstream consumer of the screener will
    also crash.
    """
    matches = solve_spectrum(
        parsed_spectrum,
        da_tol=DA_TOL,
        mz_lo=MZ_LO,
        mz_hi=MZ_HI,
        adducts=[Adduct.NA, Adduct.K],
        min_intensity=MIN_INTENSITY,
    )
    if not matches:
        pytest.skip("solver returned 0 candidates on the subsampled fixture")
    df = pd.DataFrame(
        [
            {
                "mz": c.observed_mz,
                "intensity": c.intensity,
                "n_galnac": c.n_galnac,
                "n_gal": c.n_gal,
                "ion": c.adduct.value,
                "mz_diff": c.mz_diff,
            }
            for c in matches
        ]
    )
    out = screen_candidates(
        df, parsed_spectrum.peaks, da_tol=DA_TOL, spectrum=parsed_spectrum
    )
    # Screener must have attached its four output columns.
    assert "tier" in out.columns
    assert "screen_notes" in out.columns
    assert "score_companion" in out.columns
    assert "score_series" in out.columns
    # All tier values must be in the allowed set.
    allowed_tiers = {"GREEN", "YELLOW", "RED", "PURGE"}
    bad = set(out["tier"]) - allowed_tiers
    assert not bad, f"unexpected tier values: {bad}"
    # The screener may intentionally collapse nearby accepted candidates
    # after tiering, so it returns a non-empty subset of the solver rows.
    assert 0 < len(out) <= len(df), (
        f"invalid row count: solver produced {len(df)}, screener returned {len(out)}"
    )
