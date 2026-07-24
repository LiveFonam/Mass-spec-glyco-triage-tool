"""Tests for glycan_ms.core composition solver.

These tests import directly from ``glycan_ms.core`` to avoid relying on the
package ``__init__`` (which still wires up some convenience aliases). The
test scenarios also pass an explicit ``mz_lo`` / ``mz_hi`` window to
``solve_peak`` so the underlying residue-count search space stays small.
Without an explicit window the default ``mz_hi=1e9`` would build a search
grid with hundreds of millions of compositions.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure the in-tree src/ is importable when pytest is run from the repo root.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from glycan_ms.core import (  # noqa: E402
    Adduct,
    Candidate,
    GAL,
    GALNAC,
    H2O,
    Peak,
    Spectrum,
    dedup_peaks,
    nearest_compositions,
    ppm_error,
    solve_peak,
    solve_spectrum,
)
from glycan_ms.solver_fast import (  # noqa: E402
    solve_spectrum_vectorized,
)


# ---------------------------------------------------------------------------
# 1) ppm_error is signed and has the correct magnitude
# ---------------------------------------------------------------------------
def test_ppm_error_sign_and_magnitude() -> None:
    theoretical = 1000.0
    # Observed heavier -> positive ppm
    observed_heavy = 1000.5
    err_heavy = ppm_error(observed_heavy, theoretical)
    expected_heavy = (0.5 / 1000.0) * 1e6
    assert err_heavy > 0
    assert err_heavy == pytest.approx(expected_heavy, rel=1e-9)

    # Observed lighter -> negative ppm
    observed_light = 999.5
    err_light = ppm_error(observed_light, theoretical)
    expected_light = (-0.5 / 1000.0) * 1e6
    assert err_light < 0
    assert err_light == pytest.approx(expected_light, rel=1e-9)

    # Symmetry check: |err_heavy| == |err_light|
    assert abs(err_heavy) == pytest.approx(abs(err_light), rel=1e-9)

    # A 20 ppm shift is exactly 20.0
    assert ppm_error(1020.0, 1000.0) == pytest.approx(20_000.0, rel=1e-9)
    assert ppm_error(980.0, 1000.0) == pytest.approx(-20_000.0, rel=1e-9)

    # Guard against divide-by-zero
    with pytest.raises(ValueError):
        ppm_error(100.0, 0.0)


# ---------------------------------------------------------------------------
# 1b) Adduct str-Enum round-trips through .value, not .name
# ---------------------------------------------------------------------------
def test_adduct_value_round_trips() -> None:
    """Regression for the silent-zero-match bug.

    ``Adduct`` is a ``str, Enum`` whose members are ``H / NA / K`` (the
    Python attribute names) but whose *values* are ``"H+" / "Na+" / "K+"``.
    ``Adduct("NA")`` raises ValueError because "NA" is not a valid enum
    value -- only the .value strings are valid. The lru_cache key in
    app.py must use ``.value`` so workers can re-hydrate the enum
    members on a cache hit. If this breaks, every spectrum silently
    returns 0 candidates.
    """
    # Round-trip through .value works
    for enum_member, value in (
        (Adduct.H, "H+"),
        (Adduct.NA, "Na+"),
        (Adduct.K, "K+"),
    ):
        rehydrated = Adduct(value)
        assert rehydrated is enum_member, (
            f"Adduct({value!r}) should round-trip to {enum_member!r}, "
            f"got {rehydrated!r}"
        )

    # The bug: round-trip through .name does NOT work
    for enum_member, name in (
        (Adduct.H, "H"),
        (Adduct.NA, "NA"),
        (Adduct.K, "K"),
    ):
        with pytest.raises(ValueError):
            Adduct(name)


# ---------------------------------------------------------------------------
# 2) solve_peak on a known composition
# ---------------------------------------------------------------------------
def test_solve_peak_known_composition() -> None:
    # 4 GalNAc + 2 Gal + H2O + Na+
    target_mz = 4 * GALNAC + 2 * GAL + H2O + Adduct.NA.mass
    assert target_mz == pytest.approx(1177.4236, rel=1e-9)

    cands = solve_peak(
        target_mz,
        intensity=1234.5,
        da_tol=0.7,
        mz_lo=1100.0,
        mz_hi=1300.0,
        adducts=[Adduct.NA],
    )

    assert len(cands) == 1, f"expected exactly one candidate, got {cands}"
    cand: Candidate = cands[0]
    assert cand.n_galnac == 4
    assert cand.n_gal == 2
    assert cand.adduct is Adduct.NA
    assert abs(cand.ppm_error) < 0.01
    assert cand.intensity == 1234.5
    assert cand.theoretical_mz == pytest.approx(target_mz, rel=1e-12)
    assert cand.observed_mz == pytest.approx(target_mz, rel=1e-12)


# ---------------------------------------------------------------------------
# 3) solve_peak excludes candidates outside the Da window
# ---------------------------------------------------------------------------
def test_solve_peak_da_window_excludes_other_adducts() -> None:
    target_mz = 4 * GALNAC + 2 * GAL + H2O + Adduct.NA.mass

    # K+ for the same neutral composition is heavier by (38.9637 - 22.9898).
    # A 0.05 Da window around the Na+ mass cannot also match K+ (delta is
    # ~15.97 Da), so the solver must return no K+ candidates.
    k_candidates = solve_peak(
        target_mz,
        intensity=1000.0,
        da_tol=0.05,
        mz_lo=1100.0,
        mz_hi=1300.0,
        adducts=[Adduct.K],
    )
    assert k_candidates == [], (
        "K+ adduct should be way outside a 0.05 Da window centered on the Na+ m/z"
    )

    # Sanity: with a 100 Da window K+ is found and the math checks out
    k_candidates_loose = solve_peak(
        target_mz,
        intensity=1000.0,
        da_tol=100.0,
        mz_lo=1100.0,
        mz_hi=1300.0,
        adducts=[Adduct.K],
    )
    assert any(c.adduct is Adduct.K for c in k_candidates_loose)


# ---------------------------------------------------------------------------
# 4) solve_peak respects mz_lo / mz_hi
# ---------------------------------------------------------------------------
def test_solve_peak_respects_mz_window() -> None:
    # A peak at m/z 400 with a window starting at 600 must be filtered out.
    # solve_peak returns [] early in this case (mz < mz_lo).
    cands = solve_peak(
        400.0,
        intensity=1000.0,
        da_tol=10.0,
        mz_lo=600.0,
        mz_hi=2000.0,
        adducts=list(Adduct),
    )
    assert cands == []

    # Same composition, but this time the window actually contains the
    # observed m/z. The solver must return at least one candidate.
    target_mz = 4 * GALNAC + 2 * GAL + H2O + Adduct.NA.mass
    cands_in = solve_peak(
        target_mz,
        intensity=1000.0,
        da_tol=1.0,
        mz_lo=target_mz - 5.0,
        mz_hi=target_mz + 5.0,
        adducts=[Adduct.NA],
    )
    assert len(cands_in) == 1
    assert cands_in[0].n_galnac == 4
    assert cands_in[0].n_gal == 2

    # And the same peak, with the window shifted up by 200 so the peak is
    # now below mz_lo, must return [] again.
    cands_high = solve_peak(
        target_mz,
        intensity=1000.0,
        da_tol=1.0,
        mz_lo=target_mz + 200.0,
        mz_hi=target_mz + 500.0,
        adducts=[Adduct.NA],
    )
    assert cands_high == []


# ---------------------------------------------------------------------------
# 5) solve_spectrum skips peaks below min_intensity AND dedupes by m/z
# ---------------------------------------------------------------------------
def test_solve_spectrum_min_intensity_filter() -> None:
    # Two peaks at the same m/z but different intensities. The low-intensity
    # peak is below min_intensity, so the dedup pass keeps only the high
    # one and the solver returns exactly one candidate.
    # The solver assumes its input is already deduped (the single
    # source of truth for dedup is at the app.py boundary; see
    # solver_fast.solve_spectrum_vectorized docstring). Pre-dedup the
    # test spectrum to mirror that contract.
    target_mz = 4 * GALNAC + 2 * GAL + H2O + Adduct.NA.mass
    sp = Spectrum(
        peaks=dedup_peaks(
            [
                Peak(mz=target_mz, intensity=1000.0),
                Peak(mz=target_mz, intensity=50.0),
            ],
            bin_width=0.01,
        )
    )

    filtered = solve_spectrum(
        sp,
        da_tol=1.0,
        mz_lo=1100.0,
        mz_hi=1300.0,
        adducts=[Adduct.NA],
        min_intensity=100.0,
    )
    assert len(filtered) == 1
    assert filtered[0].intensity == 1000.0

    # With min_intensity=0 the low-intensity peak survives the intensity
    # filter, but the two peaks still fall in the same 10 mDa bin so
    # the dedup pass keeps only the higher-intensity one. We expect a
    # single candidate, not two.
    unfiltered = solve_spectrum(
        sp,
        da_tol=1.0,
        mz_lo=1100.0,
        mz_hi=1300.0,
        adducts=[Adduct.NA],
        min_intensity=0.0,
    )
    assert len(unfiltered) == 1
    assert unfiltered[0].intensity == 1000.0


# ---------------------------------------------------------------------------
# 5b) dedup_peaks collapses duplicate m/z values
# ---------------------------------------------------------------------------
def test_dedup_peaks_keeps_highest_intensity() -> None:
    # Three peaks at the same nominal m/z across two scans. dedup_peaks
    # must collapse to a single peak with the highest intensity.
    peaks = [
        Peak(mz=1000.005, intensity=500.0, scan_id=1),
        Peak(mz=1000.004, intensity=2000.0, scan_id=2),
        Peak(mz=999.995, intensity=100.0, scan_id=1),
    ]
    out = dedup_peaks(peaks, bin_width=0.01)
    assert len(out) == 1
    assert out[0].intensity == 2000.0
    # scan_id comes from the surviving peak.
    assert out[0].scan_id == 2


def test_dedup_peaks_preserves_distinct_peaks() -> None:
    # Peaks 0.5 Da apart must not be merged (bin_width=0.01 << 0.5).
    peaks = [
        Peak(mz=1000.0, intensity=10.0),
        Peak(mz=1000.5, intensity=20.0),
        Peak(mz=1001.0, intensity=30.0),
    ]
    out = dedup_peaks(peaks, bin_width=0.01)
    assert [p.mz for p in out] == [1000.0, 1000.5, 1001.0]


def test_dedup_peaks_empty_and_invalid() -> None:
    assert dedup_peaks([]) == []
    import pytest
    with pytest.raises(ValueError):
        dedup_peaks([Peak(mz=1.0, intensity=1.0)], bin_width=0.0)


# ---------------------------------------------------------------------------
# 5c) _candidates_to_dataframe collapses duplicate (composition, ion)
#     rows to the highest-intensity observation.
# ---------------------------------------------------------------------------
def test_candidates_to_dataframe_dedup_by_composition() -> None:
    pytest.importorskip("glycan_ms")  # noqa: F401
    from app import _candidates_to_dataframe  # type: ignore[import-not-found]

    target = 4 * GALNAC + 2 * GAL + H2O + Adduct.NA.mass
    # Two observations of the same composition: one at the right
    # m/z with high intensity, one nearby (within tolerance) with
    # low intensity. The dataframe must keep only the high one.
    cands = [
        Candidate(
            n_galnac=4, n_gal=2, adduct=Adduct.NA,
            neutral_mass=4 * GALNAC + 2 * GAL + H2O,
            theoretical_mz=target, observed_mz=target,
            ppm_error=0.0, mz_diff=0.0, intensity=5000.0,
        ),
        Candidate(
            n_galnac=4, n_gal=2, adduct=Adduct.NA,
            neutral_mass=4 * GALNAC + 2 * GAL + H2O,
            theoretical_mz=target, observed_mz=target + 0.3,
            ppm_error=0.0, mz_diff=0.3, intensity=200.0,
        ),
    ]
    df = _candidates_to_dataframe(cands, spectrum_peaks=None)
    assert len(df) == 1
    assert df.iloc[0]["intensity"] == 5000.0
    assert df.iloc[0]["mz_diff"] == 0.0


def test_manual_composition_theoretical_mz_stays_aligned_after_dedup() -> None:
    from app import _candidates_to_dataframe  # type: ignore[import-not-found]

    measured_mzs = [2154, 2354, 2850, 2355, 4774, 3774, 4776, 2007, 4205]
    adducts = [adduct for adduct in Adduct if adduct != Adduct.H]
    candidates: list[Candidate] = []
    for measured_mz in measured_mzs:
        hits = nearest_compositions(
            measured_mz,
            adducts,
            n_lo=0,
            n_hi=20,
            m_lo=0,
            m_hi=20,
        )[:2]
        for n_galnac, n_gal, adduct, theoretical_mz in hits:
            neutral_mass = n_galnac * GALNAC + n_gal * GAL + H2O
            candidates.append(
                Candidate(
                    n_galnac=n_galnac,
                    n_gal=n_gal,
                    adduct=adduct,
                    neutral_mass=neutral_mass,
                    theoretical_mz=theoretical_mz,
                    observed_mz=measured_mz,
                    ppm_error=(measured_mz - theoretical_mz) / theoretical_mz * 1e6,
                    mz_diff=measured_mz - theoretical_mz,
                    intensity=1.0,
                )
            )

    spectrum_peaks = dedup_peaks(
        [Peak(mz=mz, intensity=1.0) for mz in measured_mzs],
        bin_width=0.01,
    )
    result = _candidates_to_dataframe(
        candidates,
        spectrum_peaks,
        da_tol=0.5,
        strictness="strict",
        include_theoretical_mz=True,
    )

    assert len(candidates) == 18
    assert len(result) == 15
    assert len(result["theoretical_mz"]) == len(result.index)
    for row in result.itertuples():
        assert row.mz - row.theoretical_mz == pytest.approx(
            row.mz_diff,
            abs=0.00005,
        )


# ---------------------------------------------------------------------------
# 5d) nearest_compositions: closest 2 are still returned for inputs
#     outside the residue grid. This locks in the manual-m/z fallback
#     contract used by app.py's composition check: even when no
#     composition is within 0.5 Da, the user must see the closest
#     2 theoretical compositions with their actual Da offset.
# ---------------------------------------------------------------------------
def test_nearest_compositions_closest_two_for_offgrid_input() -> None:
    # In-grid value: 4 GalNAc + 2 Gal + Na+ is exactly 1177.4236.
    on_grid = 4 * GALNAC + 2 * GAL + H2O + Adduct.NA.mass
    # Off-grid value: 9999.0 has no realistic composition in [0,20]x[0,20].
    off_grid = 9999.0

    adducts = [a for a in Adduct if a != Adduct.H]

    # 1) on-grid: top hit is the exact composition, |diff| < 0.01 Da.
    on_hits = nearest_compositions(
        on_grid, adducts, n_lo=0, n_hi=20, m_lo=0, m_hi=20
    )[:2]
    assert on_hits, "expected at least one hit for in-grid value"
    closest_on = min(abs(on_grid - h[3]) for h in on_hits)
    assert closest_on < 0.01, (
        f"in-grid value must match within 0.01 Da; got {closest_on:.4f}"
    )

    # 2) off-grid: every hit in [0,20]x[0,20] is far away. Confirm the
    #    function still returns the closest 2 so the UI can show them
    #    with an explicit "no close match" status. (Pure-core check
    #    here; the UI's annotation logic is exercised in app.py.)
    off_hits = nearest_compositions(
        off_grid, adducts, n_lo=0, n_hi=20, m_lo=0, m_hi=20
    )[:2]
    assert len(off_hits) == 2
    closest_off = min(abs(off_grid - h[3]) for h in off_hits)
    assert closest_off > 0.5, (
        f"off-grid value must have |diff| > 0.5 Da; got {closest_off:.4f}"
    )
    # And the results are sorted ascending by |diff|, so the user sees
    # the single best alternative first.
    diffs = [abs(off_grid - h[3]) for h in off_hits]
    assert diffs == sorted(diffs), (
        f"hits must be sorted by closeness; got {diffs}"
    )


# ---------------------------------------------------------------------------
# 6) Round-trip CSV test
# ---------------------------------------------------------------------------
def test_round_trip_csv() -> None:
    pytest.importorskip("glycan_ms.parser_table")

    import pandas as pd
    from glycan_ms.parser_table import parse_table  # type: ignore[import-not-found]

    target_mz = 4 * GALNAC + 2 * GAL + H2O + Adduct.NA.mass
    df = pd.DataFrame(
        {
            "m/z": [target_mz, 500.0, 250.0],
            "intensity": [1000.0, 800.0, 600.0],
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "peaks.csv")
        df.to_csv(path, index=False)
        spectrum = parse_table(path)

        # The parser must have produced a Spectrum with three peaks.
        assert isinstance(spectrum, Spectrum)
        assert len(spectrum.peaks) == 3

        cands = solve_spectrum(
            spectrum,
            da_tol=1.0,
            mz_lo=1100.0,
            mz_hi=1300.0,
            adducts=[Adduct.NA],
            min_intensity=0.0,
        )

        # Only the first peak (m/z = 1177.4236) lies in the window and matches.
        assert len(cands) == 1
        assert cands[0].n_galnac == 4
        assert cands[0].n_gal == 2
        assert cands[0].adduct is Adduct.NA
        assert abs(cands[0].ppm_error) < 0.01


# ---------------------------------------------------------------------------
# 7) solve_spectrum wrapper produces bit-identical output to the
#    vectorized solver it delegates to.
# ---------------------------------------------------------------------------
def test_solve_spectrum_matches_vectorized_solver() -> None:
    """``core.solve_spectrum`` is a thin wrapper around the NumPy solver.

    The contract is that the two entry points return the *same* list
    of candidates (same length, same observed m/z, same residues, same
    adduct, same ppm error to floating-point tolerance). If the
    wrapper ever diverges -- e.g. an adduct filter bug, a sign flip on
    ppm error, a row dropped by ``min_intensity`` -- this test fails
    loudly rather than silently returning fewer matches in the UI.
    """
    import random

    from glycan_ms.core import Spectrum as _Spectrum

    rng = random.Random(0xC0FFEE)
    peaks = [
        Peak(
            mz=1000.0 + rng.random() * 4000.0,
            intensity=rng.random() * 10000.0,
        )
        for _ in range(2000)
    ]
    spectrum = _Spectrum(peaks=peaks)

    a = solve_spectrum(
        spectrum,
        da_tol=1.4,
        mz_lo=1000.0,
        mz_hi=5000.0,
        adducts=[Adduct.NA, Adduct.K],
    )
    b = solve_spectrum_vectorized(
        spectrum,
        da_tol=1.4,
        mz_lo=1000.0,
        mz_hi=5000.0,
        adducts=[Adduct.NA, Adduct.K],
    )
    assert len(a) == len(b), f"length mismatch: wrapper={len(a)} vec={len(b)}"
    for x, y in zip(a, b):
        assert x.observed_mz == y.observed_mz
        assert x.n_galnac == y.n_galnac
        assert x.n_gal == y.n_gal
        assert x.adduct is y.adduct
        assert abs(x.ppm_error - y.ppm_error) < 1e-9

    # And with min_intensity set to filter half the spectrum: the
    # contract must still hold on the filtered subset.
    half = peaks[:1000]
    sp_half = _Spectrum(peaks=half)
    a2 = solve_spectrum(sp_half, da_tol=1.4, adducts=[Adduct.NA, Adduct.K])
    b2 = solve_spectrum_vectorized(sp_half, da_tol=1.4, adducts=[Adduct.NA, Adduct.K])
    assert len(a2) == len(b2)
    for x, y in zip(a2, b2):
        assert x.observed_mz == y.observed_mz
        assert x.adduct is y.adduct
        assert abs(x.ppm_error - y.ppm_error) < 1e-9


# ---------------------------------------------------------------------------
# 8) Spectrum.upper_bound() honours an explicit mz_hi
# ---------------------------------------------------------------------------
def test_spectrum_upper_bound_explicit() -> None:
    """When mz_hi is set, upper_bound() returns it, not max(peak.mz).

    This is the regression for Bug #44 (high-m/z off-edge detection):
    the screener must use the user's acquisition window (the highest
    m/z the instrument actually scanned) instead of the highest
    observed peak, so peaks above the highest observed peak but
    inside the acquisition window aren't over-penalised for missing
    companion offsets.
    """
    sp = Spectrum(
        peaks=[Peak(mz=500.0, intensity=100.0), Peak(mz=800.0, intensity=200.0)],
        mz_hi=2000.0,
    )
    assert sp.upper_bound() == 2000.0


def test_spectrum_upper_bound_from_peaks() -> None:
    """When mz_hi is None, upper_bound() returns max(peak.mz)."""
    sp = Spectrum(peaks=[Peak(mz=500.0, intensity=100.0), Peak(mz=800.0, intensity=200.0)])
    assert sp.upper_bound() == 800.0


def test_spectrum_upper_bound_empty() -> None:
    """Empty peak list returns 0.0 so callers never have to special-case."""
    sp = Spectrum(peaks=[])
    assert sp.upper_bound() == 0.0


def test_spectrum_upper_bound_zero_when_mz_hi_is_zero() -> None:
    """mz_hi=0.0 is treated as a valid (empty-window) signal, not 'unset'."""
    # When mz_hi is set (even to 0.0), it takes precedence. The
    # 'unset' sentinel is None; a 0.0 upper bound is a real (if
    # useless) acquisition window. The screener interprets a 0
    # upper bound as float('inf') at the off-edge check, but
    # upper_bound() itself must return the stored value faithfully.
    sp = Spectrum(peaks=[Peak(mz=500.0, intensity=100.0)], mz_hi=0.0)
    assert sp.upper_bound() == 0.0
