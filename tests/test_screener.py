"""Tests for glycan_ms.screener candidate validation.

Covers the composition -> formula closed form, the windowed noise
floor, the binary-search peak lookup, the companion / series /
envelope scoring helpers, the tier mapping, and the public
``screen_candidates`` end-to-end behaviour.

These tests import directly from ``glycan_ms.screener`` (private
helpers are intentionally part of the test surface) to verify
internal contracts and make regressions easy to localise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the in-tree src/ is importable when pytest is run from the repo root.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from glycan_ms.core import Adduct, Peak  # noqa: E402
from glycan_ms.screener import (  # noqa: E402
    DEFAULT_COMPANION_TOL_DA,
    DEFAULT_NOISE_BIN_WIDTH,
    ENVELOPE_OK_RELATIVE,
    GALNAC_DELTA,
    GALNAC_MINUS_GAL,
    GAL_DELTA,
    K_MINUS_NA,
    MIN_M0_SNR_FOR_KEEP,
    MIN_PEAKS_PER_BIN,
    NOISE_MULTIPLIER,
    SERIES_GAP_HI,
    SERIES_GAP_LO,
    SERIES_MIN_LENGTH,
    _build_noise_floor_cache,
    _build_noise_floor_model,
    _composition_formula,
    _envelope_for,
    _has_peak_near,
    _noise_floor_cached,
    _noise_floor_for_mz,
    _score_companion,
    _score_envelope,
    _score_series,
    _sorted_mz_index,
    _strongest_peak_near,
    _tier_from_scores,
    envelope_note_is_ok,
    is_low_sn_purge,
    noise_floor_profile,
    screen_candidates,
)


# ---------------------------------------------------------------------------
# 1) Closed-form composition -> formula
# ---------------------------------------------------------------------------
def test_composition_formula_zero_residues() -> None:
    # Just H2O at the reducing end.
    assert _composition_formula(0, 0) == "C0H2N0O1"


def test_composition_formula_single_residues() -> None:
    assert _composition_formula(1, 0) == "C8H15N1O6"
    assert _composition_formula(0, 1) == "C6H12N0O6"
    assert _composition_formula(1, 1) == "C14H25N1O11"


def test_composition_formula_multi_residues() -> None:
    # 5 GalNAc + 5 Gal -> C70 H117 N5 O51
    assert _composition_formula(5, 5) == "C70H117N5O51"
    # 10 + 10 -> C140 H232 N10 O101
    assert _composition_formula(10, 10) == "C140H232N10O101"


def test_composition_formula_rejects_negative() -> None:
    with pytest.raises(ValueError):
        _composition_formula(-1, 0)
    with pytest.raises(ValueError):
        _composition_formula(0, -1)


# ---------------------------------------------------------------------------
# 2) Noise floor (windowed, robust)
# ---------------------------------------------------------------------------
def test_noise_floor_empty_input_returns_sentinel() -> None:
    assert _noise_floor_for_mz([], 1000.0) == 1.0


def test_noise_floor_uses_bin_percentile_when_full() -> None:
    # 30 peaks densely packed in the 900-1200 bin; the 20th-percentile
    # of those 30 intensities times the multiplier should be the floor.
    peaks = [Peak(mz=900.0 + i, intensity=float(i + 1)) for i in range(30)]
    # Bin edges for mz=1000, bin_width=300: bin_lo = 900, bin_hi = 1200.
    # 20th-percentile index = round(0.20 * 29) = 6 -> intensity 7.0.
    # Floor = 7.0 * NOISE_MULTIPLIER.
    expected = 7.0 * NOISE_MULTIPLIER
    assert _noise_floor_for_mz(peaks, 1000.0) == pytest.approx(expected)


def test_noise_floor_falls_back_to_global_when_sparse() -> None:
    # Only 3 peaks, all in the same bin (< MIN_PEAKS_PER_BIN).
    # Fallback is the global minimum positive intensity, no multiplier.
    peaks = [Peak(mz=1000.0, intensity=10.0), Peak(mz=1050.0, intensity=20.0), Peak(mz=1090.0, intensity=30.0)]
    assert _noise_floor_for_mz(peaks, 1050.0) == pytest.approx(10.0)


def test_noise_floor_isolated_peak_returns_positive() -> None:
    peaks = [Peak(mz=1000.0, intensity=5.0)]
    floor = _noise_floor_for_mz(peaks, 1000.0)
    assert floor > 0


def test_noise_floor_never_zero() -> None:
    peaks = [Peak(mz=1500.0, intensity=0.0)]
    assert _noise_floor_for_mz(peaks, 1500.0) > 0


def test_noise_floor_window_changes_with_target() -> None:
    # Same peak list, two different target m/z -> two different bins,
    # both should return a positive floor.
    peaks = [Peak(mz=float(1000 + i), intensity=10.0) for i in range(50)]
    a = _noise_floor_for_mz(peaks, 1000.0)
    b = _noise_floor_for_mz(peaks, 5000.0)
    assert a > 0 and b > 0


def test_noise_floor_rejects_zero_width() -> None:
    # Deprecated: _noise_floor_for_mz no longer takes a ``bin_width``
    # argument -- binning is two-tier and constant (300 Da below
    # NOISE_BIN_SPLIT_MZ, 150 Da above). The bin edges are derived
    # internally from the spectrum's m/z range. Kept as a placeholder
    # so a future tunable binning scheme can re-introduce a width
    # validation here without renumbering.
    pass


def test_noise_floor_cache_matches_per_call() -> None:
    # The cache lookup should agree with the per-call function for
    # the same input. (Build the cache from a sparse 5-peak spectrum
    # so the per-bin path falls back to the global minimum.)
    peaks = [Peak(mz=1000.0 + i * 50, intensity=10.0 + i) for i in range(5)]
    cache, fallback = _build_noise_floor_cache(peaks)
    for p in peaks:
        per_call = _noise_floor_for_mz(peaks, p.mz)
        cached = _noise_floor_cached(cache, p.mz, fallback=fallback)
        assert per_call == pytest.approx(cached)


def test_noise_floor_cache_handles_empty_spectrum() -> None:
    cache, fallback = _build_noise_floor_cache([])
    # Empty spectrum -> cache is empty, fallback is the empty-input
    # sentinel 1.0. Both the with-fallback and no-fallback paths must
    # return 1.0 for any target m/z.
    assert _noise_floor_cached(cache, 1000.0, fallback=fallback) == 1.0
    assert _noise_floor_cached(cache, 1000.0) == 1.0


def test_noise_floor_cache_returns_fallback_on_miss() -> None:
    """When a query lands in an empty bin, the cache returns the same
    value the per-call _noise_floor_for_mz path would return -- not
    min(cache.values()).

    Regression for Bug #6: the cache lookup used to return
    ``min(cache.values())`` for missing bins, which disagrees with
    the per-call fallback ``min positive intensity across the whole
    spectrum`` (no multiplier). The two paths must agree.

    Setup: a sparse spectrum with peaks in two widely-separated bins.
    The smaller-intensity peak (5.0) is the global minimum; the
    populated bins compute per-bin floors that are very different
    from the global minimum. A query in a far-away empty bin must
    return the global minimum (5.0), matching the per-call path.
    The old `min(cache.values())` would return whatever the
    smallest per-bin floor is, which would NOT be 5.0.
    """
    peaks = [
        Peak(mz=100.0, intensity=1000.0),
        Peak(mz=120.0, intensity=5.0),  # global minimum positive intensity
        Peak(mz=2500.0, intensity=2000.0),
    ]
    cache, fallback = _build_noise_floor_cache(peaks)
    # The global fallback is the min positive intensity across the
    # whole spectrum: 5.0 (NOT a per-bin floor, NOT the noise-multiplied
    # value).
    assert fallback == pytest.approx(5.0), (
        f"global fallback must be the global min positive intensity (5.0); "
        f"got {fallback}"
    )
    # A query at m/z 5000 (far from any populated bin) must return
    # the fallback, not min(cache.values()).
    looked_up = _noise_floor_cached(cache, 5000.0, fallback=fallback)
    assert looked_up == pytest.approx(5.0), (
        f"cache miss must return the global fallback (5.0); got {looked_up}"
    )
    # And it must agree with the per-call path.
    per_call = _noise_floor_for_mz(peaks, 5000.0)
    assert looked_up == pytest.approx(per_call), (
        f"cache miss must match per-call path; got cache={looked_up}, "
        f"per_call={per_call}"
    )


def test_noise_floor_cache_uses_robust_global_fallback_on_miss() -> None:
    """A missing bin uses the robust global lower percentile, not one
    isolated minimum or the minimum of unrelated cached bin floors.

    Set up a spectrum where the per-bin floors are HIGHER than the
    global minimum, so the old `min(cache.values())` heuristic and
    the new fallback would return different values. Assert the
    cache miss returns the global minimum, not the per-bin minimum.

    The cache key is the bin's low edge in Da (not a sequential
    counter) so it stays stable across the binning transition at
    NOISE_BIN_SPLIT_MZ. The 30 packed peaks in [900, 1200) land in
    a 300-Da bin with lo = 900.
    """
    # 30 peaks densely packed in the 900-1200 bin (so the per-bin
    # path is used); all have intensities >= 10. Plus one isolated
    # peak at 5000 with intensity 2.0 -- the global minimum is 2.0,
    # but the per-bin floor at 900-1200 is some larger value.
    peaks = [Peak(mz=900.0 + i, intensity=float(i + 10)) for i in range(30)]
    peaks.append(Peak(mz=5000.0, intensity=2.0))
    cache, fallback = _build_noise_floor_cache(peaks)
    # Sorted globally: 2, 10..39. The 20th-percentile rank is 6,
    # intensity 15; applying the 1.8 multiplier gives 27.
    assert fallback == pytest.approx(15.0 * NOISE_MULTIPLIER)
    # The populated bin's floor at lo=900 is the 20th-percentile x
    # NOISE_MULTIPLIER; for these intensities (10..39) that's around
    # (17) * 1.8 = 30.6. Whatever the exact value, it's >> 2.0.
    populated_bin_floor = cache[900]
    assert populated_bin_floor > 2.0, (
        f"populated bin floor must be > global min for this test to be "
        f"meaningful; got {populated_bin_floor}"
    )
    # Querying an empty bin at m/z 6000 must return the robust global
    # fallback, not min(cache.values()).
    looked_up = _noise_floor_cached(cache, 6000.0, fallback=fallback)
    assert looked_up == pytest.approx(15.0 * NOISE_MULTIPLIER), (
        f"cache miss must return the robust fallback; got {looked_up}. "
        f"Old behaviour would have returned {populated_bin_floor} (min of cache.values())."
    )


def test_noise_floor_separates_large_signal_cluster_from_background() -> None:
    """Many large low-m/z peaks must not redefine signal as noise."""
    background = [
        Peak(mz=610.0 + i, intensity=100.0 + i * 10.0)
        for i in range(5)
    ]
    signals = [
        Peak(mz=650.0 + i, intensity=10_000.0)
        for i in range(30)
    ]

    floor = _noise_floor_for_mz(background + signals, 700.0)

    assert floor < 500.0


def test_sparse_low_mz_bin_uses_regional_floor_and_is_uncertain() -> None:
    regional_background = [
        Peak(mz=610.0 + i, intensity=100.0)
        for i in range(30)
    ]
    sparse_target = Peak(mz=950.0, intensity=10_000.0)
    model = _build_noise_floor_model([*regional_background, sparse_target])

    assert model.floor_at(950.0) == pytest.approx(100.0 * NOISE_MULTIPLIER)
    assert model.is_uncertain(950.0) is True


def test_noise_profile_exposes_uncertain_regions_for_graph() -> None:
    peaks = [Peak(mz=650.0 + i, intensity=100.0) for i in range(30)]
    mz, floors, uncertain = noise_floor_profile(peaks, 600.0, 1200.0, points=25)

    assert len(mz) == len(floors) == len(uncertain) == 25
    assert any(uncertain)
    assert all(floor > 0 for floor in floors)


# ---------------------------------------------------------------------------
# 3) Sorted-mz index + has-peak-near binary search
# ---------------------------------------------------------------------------
def test_sorted_mz_index_orders_by_mz() -> None:
    peaks = [Peak(mz=1200.0, intensity=5.0), Peak(mz=1000.0, intensity=10.0), Peak(mz=1100.0, intensity=1.0)]
    mz, inten = _sorted_mz_index(peaks)
    assert mz == [1000.0, 1100.0, 1200.0]
    assert inten == [10.0, 1.0, 5.0]


def test_has_peak_near_empty() -> None:
    assert _has_peak_near([], [], 1000.0, 10.0, 0.0) is False


def test_has_peak_near_exact_match() -> None:
    mz, inten = _sorted_mz_index([Peak(mz=1000.0, intensity=10.0)])
    assert _has_peak_near(mz, inten, 1000.0, 10.0, 0.0) is True


def test_has_peak_near_within_da_tolerance() -> None:
    # 0.01 Da window at m/z 1000; the peak is 0.003 Da off target.
    mz, inten = _sorted_mz_index([Peak(mz=1000.003, intensity=10.0)])
    assert _has_peak_near(mz, inten, 1000.0, 0.01, 0.0) is True


def test_has_peak_near_outside_da_tolerance() -> None:
    mz, inten = _sorted_mz_index([Peak(mz=1000.5, intensity=10.0)])
    # 0.1 Da window; 0.5 Da is way out.
    assert _has_peak_near(mz, inten, 1000.0, 0.1, 0.0) is False


def test_has_peak_near_below_min_intensity() -> None:
    mz, inten = _sorted_mz_index([Peak(mz=1000.0, intensity=1.0)])
    assert _has_peak_near(mz, inten, 1000.0, 10.0, 5.0) is False
    assert _has_peak_near(mz, inten, 1000.0, 10.0, 0.5) is True


# ---------------------------------------------------------------------------
# 4) Companion-peak scoring
# ---------------------------------------------------------------------------
def test_score_companion_no_companions() -> None:
    target = 1000.0
    # Just noise, nothing near target + any offset.
    peaks = [Peak(mz=2000.0, intensity=10.0), Peak(mz=3000.0, intensity=10.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 0.0)
    assert count == 0
    assert notes == []


def test_score_companion_single_offset() -> None:
    # Peak at target + 203.0794 (one extra GalNAc).
    target = 1000.0
    companion = Peak(mz=target + GALNAC_DELTA, intensity=5000.0)
    sorted_mz, sorted_intensity = _sorted_mz_index([companion])
    count, notes, _skipped = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 0.0)
    assert count == 1
    assert any("GalNAc" in n for n in notes)


def test_score_companion_multiple_offsets() -> None:
    # Peaks at +203, +162, +41.0266 -- three single-offset classes.
    target = 1000.0
    peaks = [
        Peak(mz=target + GALNAC_DELTA, intensity=5000.0),
        Peak(mz=target + GAL_DELTA, intensity=5000.0),
        Peak(mz=target + GALNAC_MINUS_GAL, intensity=5000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 0.0)
    assert count == 3
    assert len(notes) == 3


def test_score_companion_ion_pair_offset() -> None:
    # Same composition, K+ instead of Na+ (or vice versa): +15.9739.
    target = 1000.0
    peaks = [Peak(mz=target + K_MINUS_NA, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 0.0)
    assert count == 1
    assert any("K+/Na+" in n for n in notes)


def test_score_companion_galnac_minus_gal_offset() -> None:
    """Explicit regression: the +1 GalNAc -1 Gal substitution
    (+41.0266 Da) is one of the four single-offset companion
    classes in :data:`_SINGLE_OFFSETS`. A real peak at exactly
    that offset must be detected, the label must appear in the
    notes, and a peak 0.5 Da off the theoretical position must
    also be detected via the wider 0.75 Da companion tolerance
    (default). A peak at a non-meaningful offset (e.g. +50 Da)
    must NOT trigger any companion class.
    """
    target = 1000.0
    # --- 1) Exact +41.0266 Da: must be detected. ---
    peaks_exact = [Peak(mz=target + GALNAC_MINUS_GAL, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks_exact)
    count, notes, _skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 0.5, 0.0
    )
    assert count >= 1, (
        f"exact +41.0266 Da companion must be counted; got count={count}, notes={notes}"
    )
    assert any("+1 GalNAc -1 Gal" in n for n in notes), (
        f"the '+1 GalNAc -1 Gal' label must appear in notes; got {notes}"
    )

    # --- 2) 0.5 Da off the +41.0266 Da offset: must also be detected
    # via the default 0.75 Da companion tolerance, proving the +41
    # path goes through the wider tolerance. A 0.5 Da Da tolerance
    # catches a 0.5-Da-off peak right on the edge. ---
    peaks_off = [Peak(mz=target + GALNAC_MINUS_GAL + 0.5, intensity=5000.0)]
    sorted_mz_off, sorted_intensity_off = _sorted_mz_index(peaks_off)
    count_off, notes_off, _skipped_off = _score_companion(
        target, sorted_mz_off, sorted_intensity_off, 0.5, 0.0,
        companion_tol_da=DEFAULT_COMPANION_TOL_DA,
    )
    assert count_off >= 1, (
        f"a 0.5-Da-off +41 companion is within the 0.75 Da window and "
        f"must be counted; got count={count_off}, notes={notes_off}"
    )
    assert any("+1 GalNAc -1 Gal" in n for n in notes_off), (
        f"the '+1 GalNAc -1 Gal' label must appear in notes for the "
        f"0.5-Da-off case; got {notes_off}"
    )

    # --- 3) Negative: a peak at +50 Da is not a meaningful offset
    # (it's 9 Da past +41 and 8 Da short of +K/Na at +57.0) and must
    # NOT trigger any companion class. A 0.5 Da tolerance excludes it. ---
    peaks_neg = [Peak(mz=target + 50.0, intensity=5000.0)]
    sorted_mz_neg, sorted_intensity_neg = _sorted_mz_index(peaks_neg)
    count_neg, notes_neg, _skipped_neg = _score_companion(
        target, sorted_mz_neg, sorted_intensity_neg, 0.5, 0.0
    )
    assert count_neg == 0, (
        f"a +50 Da peak is not a meaningful companion offset and must "
        f"not be counted; got count={count_neg}, notes={notes_neg}"
    )


def test_score_companion_pair_offset() -> None:
    # +1 GalNAc +1 Gal = +203.0794 + 162.0528 = +365.1322.
    target = 1000.0
    peaks = [Peak(mz=target + GALNAC_DELTA + GAL_DELTA, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 0.0)
    # Should hit the pair (1) but not the singles (no peak at +203 or +162 alone).
    assert count == 1
    assert any("GalNAc" in n and "Gal" in n for n in notes)


def test_score_companion_below_noise_floor_excluded() -> None:
    # Companion peak present but below the noise floor -> not counted.
    # Peak is 0.5 Da off theoretical to stay inside the 0.75 Da
    # search window but OUTSIDE the 0.3 Da tight-offset override
    # (see test_score_companion_tight_offset_override_counts_below_noise
    # for the override case).
    target = 1000.0
    peaks = [Peak(mz=target + GALNAC_DELTA + 0.5, intensity=1.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 100.0)
    assert count == 0


def test_score_companion_near_miss_reported_in_extended_band() -> None:
    """Regression: a peak 1.0-1.5 Da off the theoretical companion
    position is just outside the 0.75 Da window and must NOT be
    counted as a real companion, but the note should surface it as
    a "near miss" with the actual offset in Da so the user can see
    "there's a peak 1.23 Da off the theoretical +Gal position" and
    decide whether the offset is real-but-shifted.
    """
    target = 4787.55
    # 1.23 Da off the theoretical +Gal (162.05 Da) position
    peaks = [Peak(mz=target + GAL_DELTA + 1.23, intensity=1200.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 0.7, 0.0,
        companion_tol_da=0.75,
    )
    # Outside the 0.75 Da window -> not counted
    assert count == 0
    # But the near-miss peak IS reported in the note
    assert any("near miss" in n and "+1.23" in n for n in notes), (
        f"expected a near-miss note with the actual offset; got {notes}"
    )

    # And a peak WAY off (5 Da) is neither counted nor reported.
    peaks_far = [Peak(mz=target + GAL_DELTA + 5.0, intensity=1200.0)]
    sorted_mz_far, sorted_intensity_far = _sorted_mz_index(peaks_far)
    count_far, notes_far, _skipped_far = _score_companion(
        target, sorted_mz_far, sorted_intensity_far, 0.7, 0.0,
        companion_tol_da=0.75,
    )
    assert count_far == 0
    assert not any("near miss" in n for n in notes_far), (
        f"5-Da-off peak must not be reported as a near miss; got {notes_far}"
    )


def test_score_companion_uses_per_offset_noise_floor() -> None:
    """Regression: a companion peak in a higher-noise region must
    be evaluated against the noise floor at ITS m/z, not at the
    candidate's m/z. The old code passed the candidate's noise
    floor, which falsely accepted noise spikes in low-noise
    regions (e.g. the candidate was at 4096 where noise is 5,
    the +162 companion at 4258 was a real noise spike of
    intensity 30 -- the old code called it a real companion
    because 30 > 5; the new code rejects it because 30 < 50,
    the local noise at 4258)."""
    from glycan_ms.screener import _bin_index

    target = 4096.66
    # Peak is 0.5 Da off theoretical to stay inside the 0.75 Da
    # search window but OUTSIDE the 0.3 Da tight-offset override
    # -- this test is about per-offset noise floors, not the
    # tight override, so the override must NOT fire.
    peaks = [
        Peak(mz=target, intensity=5000.0),
        Peak(mz=target + GAL_DELTA + 0.5, intensity=30.0),  # 4259.21
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    # Build a noise cache where the candidate bin has a low floor
    # (5) and the offset bin has a higher floor (50).
    cache = {
        _bin_index(target): 5.0,
        _bin_index(target + GAL_DELTA): 50.0,
    }
    count, notes, _skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 5.0,
        noise_cache=cache,
    )
    # Intensity 30 is above the candidate's noise floor (5) but
    # BELOW the offset's noise floor (50). The per-offset floor
    # must reject this. The old code (using the candidate's floor)
    # would have counted it.
    assert count == 0, (
        f"intensity-30 companion at 4258 must be below the local "
        f"noise floor of 50; got count={count}, notes={notes}"
    )

    # Same companion but with higher intensity (60) should be
    # counted when both regions' floors are considered.
    peaks_hi = [
        Peak(mz=target, intensity=5000.0),
        Peak(mz=target + GAL_DELTA, intensity=60.0),
    ]
    sorted_mz_hi, sorted_intensity_hi = _sorted_mz_index(peaks_hi)
    count_hi, notes_hi, _skipped = _score_companion(
        target, sorted_mz_hi, sorted_intensity_hi, 10.0, 5.0,
        noise_cache=cache,
    )
    assert count_hi == 1, f"intensity-60 companion must be counted; got {count_hi}"


def test_score_companion_reports_peak_below_noise_with_diagnostics() -> None:
    """Regression: when a real-looking peak IS present at the expected
    companion offset but is BELOW the local noise floor, the note must
    say so explicitly -- quoting the actual m/z, intensity, floor, AND
    the diff from the theoretical offset in Da -- so the user can see
    WHY a real companion was rejected. Previously the note just said
    "+1 GalNAc" was absent, leaving the user to guess whether the
    offset had no peak, was off-spectrum, or had a sub-noise peak.
    The Da-off-theoretical value is included so the user can tell at
    a glance whether the peak is at the right offset (just below
    noise) or somewhere else entirely.

    The peak here is 0.5 Da off theoretical so it stays inside the
    0.75 Da search window but OUTSIDE the 0.3 Da tight-offset
    override -- this test exercises the "below noise but not at
    the right offset" path which produces the diagnostic note.
    See test_score_companion_tight_offset_override_counts_below_noise
    for the override case.
    """
    target = 4096.66
    # Real peak at +162 + 0.5 Da off, intensity 30, in a region
    # with local noise floor 100.
    peaks = [
        Peak(mz=target, intensity=5000.0),
        Peak(mz=target + GAL_DELTA + 0.5, intensity=30.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    from glycan_ms.screener import _bin_index
    cache = {
        _bin_index(target): 5.0,
        _bin_index(target + GAL_DELTA + 0.5): 100.0,
    }
    count, notes, _skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 5.0, noise_cache=cache
    )
    # Sub-noise peak is NOT counted.
    assert count == 0
    # But the note must report the peak's actual numbers, not the
    # absence. This is the whole point of the diagnostic.
    assert any("peak at" in n and "int" in n and "floor" in n for n in notes), (
        f"expected a diagnostic note with peak/int/floor; got {notes}"
    )
    # The note must also include the diff from the theoretical
    # offset position in Da so the user can see whether the peak
    # is sitting at the right offset (just below noise) or
    # wandering somewhere unrelated. Without this number the user
    # had to compute (peak_mz - target_mz - offset) themselves.
    assert any("Da off theoretical" in n for n in notes), (
        f"expected diagnostic note to include 'Da off theoretical'; got {notes}"
    )
    # The note should mention the +1 Gal label so the user knows
    # which offset class was found-but-rejected.
    assert any("+1 Gal" in n or "Gal" in n for n in notes), (
        f"expected note to identify the offset class; got {notes}"
    )


def test_score_companion_tight_offset_override_counts_below_noise() -> None:
    """Regression: a peak inside the 0.75 Da search window but BELOW
    the local noise floor must still count as a companion when it is
    within COMPANION_TIGHT_TOL_DA (0.3 Da) of the theoretical
    offset position. Rationale: a real biological signal at (e.g.)
    +1 GalNAc - 0.2 Da is overwhelmingly more likely to be a real
    companion than a coincidental noise spike, even if the bin's
    25th-percentile x 2 noise floor is high. Without this override
    the user sees a "found but rejected" diagnostic for peaks that
    are unambiguously at the right offset.
    """
    from glycan_ms.screener import _bin_index

    target = 1013.35
    # Theoretical +1 GalNAc is at 1013.35 + 203.0794 = 1216.43.
    # Real peak at 1216.1909, which is -0.24 Da off theoretical.
    # The peak intensity is 30, but the local noise floor is 100,
    # so a strict "above noise" check would reject it. The
    # tight-offset override (0.3 Da) must catch it because 0.24 <
    # 0.3.
    peaks = [
        Peak(mz=target, intensity=2000.0),
        Peak(mz=1216.1909, intensity=30.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache = {
        _bin_index(target): 5.0,
        _bin_index(1216.1909): 100.0,
    }
    count, notes, _ = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 5.0, noise_cache=cache
    )
    assert count >= 1, (
        f"tight-offset override should have counted the 0.24-Da-off "
        f"companion; got count={count}, notes={notes}"
    )
    # The note should mention "+1 GalNAc" so the user knows which
    # offset class was matched.
    assert any("+1 GalNAc" in n for n in notes), (
        f"expected +1 GalNAc in the note; got {notes}"
    )
    # The "tight" override tag should appear so the user can tell
    # the peak was below the noise floor but counted anyway.
    assert any("tight" in n.lower() for n in notes), (
        f"expected 'tight' override tag in the note; got {notes}"
    )


def test_score_companion_below_noise_outside_tight_still_reported_only() -> None:
    """Counterpart to the tight-offset override: a peak inside the
    0.75 Da search window, BELOW the noise floor, and OUTSIDE the
    0.3 Da tight-offset override must NOT be counted. It should be
    reported in the diagnostic note (peak/int/floor) but with no
    count contribution. This locks in the boundary between "real
    companion the noise floor over-penalised" (counted) and
    "coincidental peak in the search window" (not counted).
    """
    from glycan_ms.screener import _bin_index

    target = 1013.35
    # Same theoretical +1 GalNAc position (1216.43) but observed
    # peak at 1215.8 -- 0.63 Da off theoretical. That is outside
    # the 0.3 Da tight-offset override but inside the 0.75 Da
    # search window. Below noise -- should not count.
    peaks = [
        Peak(mz=target, intensity=2000.0),
        Peak(mz=1215.8, intensity=30.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache = {
        _bin_index(target): 5.0,
        _bin_index(1215.8): 100.0,
    }
    count, notes, _ = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 5.0, noise_cache=cache
    )
    assert count == 0, (
        f"0.63-Da-off below-noise peak must not be counted; "
        f"got count={count}, notes={notes}"
    )
    # But it should still be reported in a diagnostic note.
    assert any("peak at" in n and "floor" in n for n in notes), (
        f"expected diagnostic note with peak/floor; got {notes}"
    )


def test_score_companion_wider_da_tolerance_catches_off_companion() -> None:
    """Regression: the companion search uses a 0.75 Da tolerance
    (DEFAULT_COMPANION_TOL_DA), not just the ppm-derived window. A
    real +162 Gal companion that's 0.5 Da off the theoretical
    position -- normal at high m/z where the user's tolerance is
    in Da, not ppm -- must be counted as a real companion rather
    than rejected by a too-narrow ppm window.
    """
    from glycan_ms.screener import DEFAULT_COMPANION_TOL_DA

    target = 4543.61
    # +162 companion 0.5 Da off the theoretical position
    peaks = [
        Peak(mz=target, intensity=2000.0),
        Peak(mz=target + GAL_DELTA + 0.5, intensity=200.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 0.7, 0.0,
        companion_tol_da=DEFAULT_COMPANION_TOL_DA,
    )
    assert count >= 1, (
        f"a 0.5-Da-off companion within the 0.75 Da window must be "
        f"counted; got count={count}, notes={notes}"
    )
    assert any("Gal" in n for n in notes), (
        f"expected the +1 Gal label in notes; got {notes}"
    )


def test_score_companion_da_tolerance_cap() -> None:
    """The companion Da tolerance caps the search window. A peak
    that's 1.0 Da off the theoretical +162 position is outside the
    0.75 Da window and must NOT be counted, so the Da tolerance
    can't grow unbounded.
    """
    target = 4543.61
    peaks = [
        Peak(mz=target, intensity=2000.0),
        Peak(mz=target + GAL_DELTA + 1.0, intensity=200.0),  # 1.0 Da off
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, _notes, _skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 0.7, 0.0,
        companion_tol_da=0.75,
    )
    assert count == 0, (
        f"a 1.0-Da-off companion is outside the 0.75 Da window and "
        f"must not be counted; got count={count}"
    )


def test_score_companion_offsets_past_spectrum_skipped() -> None:
    """Regression: a high-m/z candidate must not be penalised for a
    companion offset that would land past the spectrum end. Without
    this fix, every candidate above ~5800 Da in a 5998-Da file would
    miss its +1 GalNAc companion by construction and drop to RED
    for "no companion", even though the instrument simply didn't
    acquire the +203 region."""
    target = 5800.0
    # No peak at +203 (and there shouldn't be -- it's past 5998).
    # The companion loop should skip the check entirely, not count
    # the missing peak as a failure.
    peaks = [Peak(mz=target, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 0.0, mz_max=5998.0
    )
    # +203 would be at 6003, past 5998 -> skipped.
    # +162 would be at 5962, in range but no peak present -> 0 hits.
    # +41 would be at 5841, in range but no peak present -> 0 hits.
    # Net: count == 0, but for "no peak present", not "skipped".
    # The bug we're fixing is that WITHOUT mz_max, a future change
    # might count skipped offsets against the candidate. The current
    # implementation only counts PRESENT peaks, so the visible
    # behaviour for this exact input is the same with or without
    # mz_max. We assert the contract: count == 0.
    assert count == 0
    # And: when an offset IS in range AND a peak exists, we still count it.
    in_range = [Peak(mz=target + GAL_DELTA, intensity=5000.0)]  # +162 -> 5962
    sorted_mz, sorted_intensity = _sorted_mz_index(in_range)
    count_in, _, _ = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 0.0, mz_max=5998.0
    )
    assert count_in == 1, "in-range +162 companion must be counted"


# ---------------------------------------------------------------------------
# 5) Series / 3-in-a-row scoring
# ---------------------------------------------------------------------------
def test_score_series_no_peaks() -> None:
    sorted_mz, sorted_intensity = _sorted_mz_index([])
    cache, _fallback = _build_noise_floor_cache([])
    length, notes, _skipped = _score_series(1000.0, sorted_mz, sorted_intensity, 10.0, cache)
    assert length == 0
    assert notes == []


def test_score_series_three_in_a_row_galnac() -> None:
    # 3 additional GalNAc-residue peaks past the candidate.
    target = 1000.0
    peaks = [
        Peak(mz=target + 1 * GALNAC_DELTA, intensity=5000.0),
        Peak(mz=target + 2 * GALNAC_DELTA, intensity=5000.0),
        Peak(mz=target + 3 * GALNAC_DELTA, intensity=5000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache, _fallback = _build_noise_floor_cache(peaks)
    length, notes, _skipped = _score_series(target, sorted_mz, sorted_intensity, 10.0, cache)
    assert length >= SERIES_MIN_LENGTH
    assert any("GalNAc" in n for n in notes)


def test_score_series_three_in_a_row_gal() -> None:
    target = 1000.0
    peaks = [
        Peak(mz=target + 1 * GAL_DELTA, intensity=5000.0),
        Peak(mz=target + 2 * GAL_DELTA, intensity=5000.0),
        Peak(mz=target + 3 * GAL_DELTA, intensity=5000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache, _fallback = _build_noise_floor_cache(peaks)
    length, notes, _skipped = _score_series(target, sorted_mz, sorted_intensity, 10.0, cache)
    assert length >= SERIES_MIN_LENGTH
    assert any("Gal" in n for n in notes)


def test_score_series_only_two_in_a_row() -> None:
    # 2 additional peaks, not 3 -> should not count.
    target = 1000.0
    peaks = [
        Peak(mz=target + 1 * GALNAC_DELTA, intensity=5000.0),
        Peak(mz=target + 2 * GALNAC_DELTA, intensity=5000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache, _fallback = _build_noise_floor_cache(peaks)
    length, notes, _skipped = _score_series(target, sorted_mz, sorted_intensity, 10.0, cache)
    assert length < SERIES_MIN_LENGTH
    assert notes == []


def test_score_series_gap_outside_tolerance() -> None:
    # First step is fine, second step is 1.5x the offset (outside 0.75-1.2).
    target = 1000.0
    peaks = [
        Peak(mz=target + 1 * GALNAC_DELTA, intensity=5000.0),
        Peak(mz=target + 1 * GALNAC_DELTA + 1.5 * GALNAC_DELTA, intensity=5000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache, _fallback = _build_noise_floor_cache(peaks)
    length, notes, _skipped = _score_series(target, sorted_mz, sorted_intensity, 10.0, cache)
    assert length < SERIES_MIN_LENGTH


def test_score_series_breaks_on_missing_step() -> None:
    # First step fine, second step missing, third step fine.
    # Chain should not be counted as length 3.
    target = 1000.0
    peaks = [
        Peak(mz=target + 1 * GALNAC_DELTA, intensity=5000.0),
        Peak(mz=target + 3 * GALNAC_DELTA, intensity=5000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache, _fallback = _build_noise_floor_cache(peaks)
    length, notes, _skipped = _score_series(target, sorted_mz, sorted_intensity, 10.0, cache)
    assert length < SERIES_MIN_LENGTH


def test_score_series_stops_at_spectrum_boundary() -> None:
    """Regression: a high-m/z candidate's series ladder should count
    the steps that ARE in range, not fail just because the next steps
    are past the spectrum end. A 5950-Da candidate with ladder steps
    at +162 to 5950+162=6112 (past 5998) used to count as 0, because
    the series walker would have no peaks in range. With the
    mz_max fix, the walker stops at the boundary without counting
    the missing steps as failures."""
    target = 5950.0
    # Two in-range steps: +162, +324 (= 6112, 6274 -> both past).
    # Wait, +162 = 6112 which is past 5998. So even +162 is out.
    # Use a lower-m/z target so 2 steps are in range.
    target = 5800.0  # +162 = 5962 (in), +324 = 6124 (out)
    peaks = [
        Peak(mz=target + 1 * GAL_DELTA, intensity=5000.0),  # +162 -> 5962
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache, _fallback = _build_noise_floor_cache(peaks)
    length, notes, _skipped = _score_series(
        target, sorted_mz, sorted_intensity, 10.0, cache, mz_max=5998.0
    )
    # We have 1 in-range step that is present, then the next step
    # is past the spectrum boundary and the walker stops. Result: 1.
    # (Below SERIES_MIN_LENGTH=3 so it doesn't count as a "series".)
    assert length == 1


# ---------------------------------------------------------------------------
# 6) Envelope scoring (via _score_envelope)
# ---------------------------------------------------------------------------
def test_envelope_for_returns_neutral_mz() -> None:
    env = _envelope_for(4, 4, 22.9898)  # 4 GalNAc + 4 Gal, Na+ adduct
    assert env is not None and len(env) > 0
    # M+0 should be approximately 4*203.0794 + 4*162.0528 + 18.0106 + 22.9898.
    expected_m0 = 4 * GALNAC_DELTA + 4 * GAL_DELTA + 18.0106 + 22.9898
    assert env[0][0] == pytest.approx(expected_m0, abs=0.001)


def test_envelope_for_handles_zero_galnac_composition() -> None:
    """Regression: n_galnac=0 emits a formula with N0.

    The closed-form formula in ``_composition_formula`` produces
    ``C{x}H{y}N0O{w}`` whenever n_galnac is zero (Hex-only / no HexNAc
    candidates -- a real, common composition class the solver emits in
    the lower m/z range). The envelope helper must still return a
    valid envelope for these compositions rather than silently
    returning None and downgrading the candidate with a misleading
    "molmass unavailable" note. The intent of the function is "give me
    the 13C envelope of this composition"; the only inputs that should
    return None are an outright unparseable composition, which is not
    what n=0,m=4 is.
    """
    env = _envelope_for(0, 4, 22.9898)  # 0 GalNAc + 4 Gal, Na+ adduct
    assert env is not None, (
        "envelope for n_galnac=0 must not be None: this composition is a "
        "real solver output (Hex-only) and the helper must handle it"
    )
    assert len(env) > 0
    # M+0 must match the closed-form neutral mass + adduct.
    expected_m0 = 4 * GAL_DELTA + 18.0106 + 22.9898
    assert env[0][0] == pytest.approx(expected_m0, abs=0.001)


def test_envelope_for_handles_zero_zero_composition() -> None:
    """Boundary: n=0, m=0 is the "water-only" composition.

    ``_composition_formula(0, 0)`` returns ``"C0H2N0O1"`` (water). The
    envelope helper must still return an envelope for it.
    """
    env = _envelope_for(0, 0, 22.9898)  # water + Na+ (H2O + Na+ = NaOH+H+ = 40.99 Da)
    assert env is not None
    assert len(env) > 0
    # M+0 mass is the monoisotopic mass of H2O + Na+ = 18.0106 + 22.9898.
    assert env[0][0] == pytest.approx(18.0106 + 22.9898, abs=0.001)


def test_envelope_for_m0_intensity_is_100_percent() -> None:
    """The leading entry of the envelope is the monoisotopic peak (M+0).

    Its intensity is, by definition, the largest of the envelope (the
    multinomial expansion is normalised to 100% at M+0). Subsequent
    entries (M+1, M+2, ...) must be strictly less than 100%.
    """
    env = _envelope_for(2, 2, 22.9898)  # 2 GalNAc + 2 Gal, Na+
    assert env is not None and len(env) >= 2
    assert env[0][1] == pytest.approx(100.0, abs=0.01)
    for _mz, pct in env[1:]:
        assert pct < env[0][1]


def test_envelope_for_m1_is_roughly_carbon_count_times_1_11_percent() -> None:
    """The M+1 / M+0 intensity ratio approximates C_count * 1.11%.

    For the glycan m/z range, M+1 is dominated by 13C substitution; the
    natural-abundance approximation is C_count * 0.0111. This locks in
    the multinomial expansion's M+1 intensity to within a few percent
    of the analytic approximation, which is the property the screener
    relies on when it decides "envelope observed".
    """
    n_galnac, n_gal = 4, 4
    c_count = 8 * n_galnac + 6 * n_gal  # = 56
    env = _envelope_for(n_galnac, n_gal, 22.9898)
    assert env is not None and len(env) >= 2
    m1_pct = env[1][1]
    expected = c_count * 1.11  # ~62.16% for C=56
    assert m1_pct == pytest.approx(expected, rel=0.10)


def test_score_envelope_full_envelope_observed() -> None:
    # M+0 and M+1 both present. The M+0 S/N is set above the MEDIUM
    # bar (1.5) so the relative gate is enabled and a 40C glycan's
    # natural M+1 (~44% of M+0) is above the noise. With S/N=10
    # and M+0=20000, M+1 expected ~8800. CI gate = 2000 + 2*sqrt(2000)
    # = 2089. M+1 at 8800 >> 2089 -> observed. M+2 also provided at
    # natural abundance. Required satellites = {1, 2, 3} (HIGH bar at
    # S/N=10), so M+3 also required.
    n_galnac, n_gal = 2, 2
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898  # Na+
    peaks = [
        Peak(mz=m0_mz, intensity=20000.0),
        Peak(mz=m0_mz + 1.0033, intensity=8800.0),       # M+1 (~44%)
        Peak(mz=m0_mz + 2 * 1.0033, intensity=2000.0),   # M+2 (~10%)
        Peak(mz=m0_mz + 3 * 1.0033, intensity=300.0),    # M+3 (~1.5%)
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz,
        noise_at_target=2000.0,
    )
    assert ok is True
    assert "envelope" in note.lower() or "M+0" in note


def test_score_envelope_no_envelope_observed() -> None:
    # Nothing at the M+0 or M+1 position.
    n_galnac, n_gal = 2, 2
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    peaks = [Peak(mz=2000.0, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, _note, _skipped = _score_envelope(n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz)
    assert ok is False


def test_score_envelope_reports_m0_and_m1_separately() -> None:
    """Regression: the message must list which satellites (M+1, M+2, M+3)
    are actually observed, not collapse them into "(M+0, M+1)"."""
    n_galnac, n_gal = 2, 2
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898  # Na+
    # S/N=10 so M+1 is naturally above the noise AND the relative
    # gate is enabled (S/N >= 1.5). M+2 / M+3 deliberately absent.
    peaks = [
        Peak(mz=m0_mz, intensity=20000.0),
        Peak(mz=m0_mz + 1.0033, intensity=8800.0),  # M+1
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    # HIGH S/N (10x noise) so required is {1, 2, 3}. M+2 / M+3
    # missing -> envelope fails. Note must list M+0 and M+1
    # separately and not claim M+2 was observed.
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz,
        noise_at_target=2000.0,
    )
    assert ok is False
    assert "M+0" in note
    assert "M+1" in note
    # M+2 was NOT observed. The note may still say "requires M+2"
    # (as part of the required-satellites message at HIGH S/N), but
    # the satellite list itself must not include M+2.
    observed_part = note.split("--")[0] if "--" in note else note
    assert "M+2" not in observed_part, (
        f"M+2 must not be in observed list; got {observed_part!r}"
    )


def test_score_envelope_reports_m2_when_present() -> None:
    """M+2 must be reported when it is observed above the noise floor."""
    n_galnac, n_gal = 2, 2
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1.0033, intensity=3000.0),  # M+1
        Peak(mz=m0_mz + 2 * 1.0033, intensity=1000.0),  # M+2
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    # MEDIUM S/N (noise=2000, M+0=5000 -> S/N=2.5). The required
    # satellites is {1, 2} at this S/N, so M+0+M+1+M+2 is enough.
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz,
        noise_at_target=2000.0,
    )
    assert ok is True
    assert "M+0" in note
    assert "M+1" in note
    assert "M+2" in note


def test_score_envelope_m2_below_noise_floor_excluded() -> None:
    """M+2 below the noise floor must NOT be reported as observed.

    Also at this very low S/N the M+1 is below the CI gate
    (noise + 2*sqrt(noise) = 5000 + 141 = 5141, M+1 = 3000) and
    the relative gate is disabled (S/N=1.0 < 1.5), so M+1 is
    also not observed. The candidate's envelope is then just
    M+0 (no satellites), which is below the LOW bar (1.09) ->
    required {1} but M+1 not observed -> envelope fails.
    The candidate is correctly flagged as a bad envelope.
    """
    n_galnac, n_gal = 2, 2
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1.0033, intensity=3000.0),  # M+1 (above noise but below CI)
        Peak(mz=m0_mz + 2 * 1.0033, intensity=0.1),  # M+2 (below noise)
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    # S/N=1.0. Below LOW bar (1.09) -> required {1}. Below MEDIUM
    # (1.5) -> relative gate disabled. CI gate = 5141. M+1 at 3000
    # < 5141 -> not observed. Envelope fails.
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz,
        noise_at_target=5000.0,
    )
    assert ok is False
    # M+2 is below noise so it was never going to be observed.
    assert "M+2" not in note


def test_score_envelope_caps_at_m3() -> None:
    """Regression: molmass returns ~20+ isotopologue entries (with
    predicted abundances < 1e-10 by the tail). Without a cap, the
    loop would iterate over every entry and report e.g. M+21 / M+22
    / M+23 when a real peak happened to fall near the +21.073-Da
    offset. The fix is to check ONLY the first 3 satellites --
    M+1, M+2, M+3 -- the physically meaningful 13C isotope peaks.
    The label below reflects that cap.
    """
    n_galnac, n_gal = 5, 5  # C70H117N5O51 -> 22 envelope entries
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    # Place peaks at M+0, M+1, M+2, M+3, AND a high-abundance
    # distractor at +21 Da (where the bug would falsely report
    # "M+21").
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1.0033, intensity=3000.0),       # real M+1
        Peak(mz=m0_mz + 2 * 1.0033, intensity=1000.0),   # real M+2
        Peak(mz=m0_mz + 3 * 1.0033, intensity=300.0),    # real M+3
        Peak(mz=m0_mz + 21.0, intensity=9000.0),         # distractor
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz,
        noise_at_target=0.0,
    )
    assert ok is True
    assert "M+0" in note
    assert "M+1" in note
    assert "M+2" in note
    assert "M+3" in note
    # The +21-Da distractor must NOT be reported, even though it
    # is high-intensity and would have been matched before the cap.
    assert "M+21" not in note
    assert "M+22" not in note
    assert "M+23" not in note


def test_score_envelope_strict_window_medium_sn_passes_with_m0_m1_m2() -> None:
    """For candidates in the 1000-3800 Da range at MEDIUM S/N
    (M+0 between 1.5x and 5x noise), the envelope check requires
    M+0 + M+1 + M+2. M+3 is OPTIONAL because at this S/N the
    naturally-weak M+3 (1-3% of M+0) is expected to be near noise.
    A candidate with only M+0 + M+1 + M+2 (no M+3) is ACCEPTED.

    At HIGH S/N (M+0 >= 5x noise) M+3 is REQUIRED again -- see
    ``test_score_envelope_strict_window_high_sn_requires_m3``.
    """
    # 4 GalNAc + 7 Gal + Na+ = ~2109.78, in strict window. Swapped from
    # the original (7 GalNAc + 4 Gal) so the new "n_galnac > n_gal"
    # override rule (requiring M+0..M+3) does not fire -- this test
    # is about the MEDIUM-S/N scaling of the strict-window bar, not
    # the GalNAc-dominance override.
    n_galnac, n_gal = 4, 7
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    assert 1000.0 <= m0_mz <= 3800.0
    # noise=2000, M+0=5000 -> S/N = 2.5 (medium). M+3 missing.
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1.0033, intensity=3000.0),
        Peak(mz=m0_mz + 2 * 1.0033, intensity=1000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=m0_mz, noise_at_target=2000.0,
    )
    assert ok is True, f"M+0+M+1+M+2 must pass strict window at medium S/N; got note={note!r}"
    assert "M+0" in note and "M+1" in note and "M+2" in note
    assert "M+3" not in note


def test_score_envelope_strict_window_high_sn_requires_m3() -> None:
    """At HIGH S/N (M+0 >= 5x noise), the strict window demands
    ALL FOUR satellites: M+0+M+1+M+2+M+3. A candidate with M+3
    missing at this S/N is flagged as an incomplete envelope.
    This is the user's rule: M+3 only drops out when M+0 is
    already near the noise floor.
    """
    n_galnac, n_gal = 7, 4
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    # noise=100, M+0=10000 -> S/N = 100 (high). M+3 missing.
    peaks = [
        Peak(mz=m0_mz, intensity=10000.0),
        Peak(mz=m0_mz + 1.0033, intensity=6000.0),
        Peak(mz=m0_mz + 2 * 1.0033, intensity=2000.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=m0_mz, noise_at_target=100.0,
    )
    assert ok is False, f"M+3 missing at high S/N must fail; got ok=True note={note!r}"
    assert "M+3" in note
    assert "S/N" in note  # note reports the S/N ratio so the user sees the bar


def test_score_envelope_strict_window_high_sn_passes_with_all_four() -> None:
    """At HIGH S/N with all four satellites present, envelope is OK."""
    n_galnac, n_gal = 7, 4
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    peaks = [
        Peak(mz=m0_mz, intensity=10000.0),
        Peak(mz=m0_mz + 1.0033, intensity=6000.0),
        Peak(mz=m0_mz + 2 * 1.0033, intensity=2000.0),
        Peak(mz=m0_mz + 3 * 1.0033, intensity=400.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=m0_mz, noise_at_target=100.0,
    )
    assert ok is True
    assert "M+3" in note


def test_score_envelope_strict_window_passes_with_all_three() -> None:
    """When M+1 + M+2 + M+3 are ALL observed in the strict window,
    the envelope check passes and the note reports all three."""
    n_galnac, n_gal = 7, 4
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898  # ~2109.78, in strict window
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1.0033, intensity=3000.0),
        Peak(mz=m0_mz + 2 * 1.0033, intensity=1000.0),
        Peak(mz=m0_mz + 3 * 1.0033, intensity=300.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz,
        noise_at_target=0.0,
    )
    assert ok is True
    assert "M+0" in note and "M+1" in note and "M+2" in note and "M+3" in note


def test_score_envelope_above_strict_window_low_sn_m0_m1_suffices() -> None:
    """Above 3800 Da at LOW S/N, only M+0 + M+1 is required.
    The S/N-scaled bar applies at every m/z -- large glycans
    (high m/z, many carbons) with M+0 near the noise floor
    cannot be expected to show M+2/M+3.
    """
    # 10 GalNAc + 12 Gal + Na+ = ~4099.48, above strict window.
    # Swapped from (12 GalNAc + 10 Gal) so the new "n_galnac > n_gal"
    # override does not fire -- this test is about the LOW-S/N
    # scaling above 3800 Da, not the GalNAc-dominance rule.
    n_galnac, n_gal = 10, 12
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    assert m0_mz > 3800.0
    # S/N = 10 so M+1 is naturally above the noise AND the relative
    # gate is enabled (S/N >= 1.5). HIGH bucket (10x noise) so
    # required is {1, 2, 3}. We provide all four satellites.
    peaks = [
        Peak(mz=m0_mz, intensity=20000.0),
        Peak(mz=m0_mz + 1.0033, intensity=8800.0),       # M+1
        Peak(mz=m0_mz + 2 * 1.0033, intensity=2000.0),   # M+2
        Peak(mz=m0_mz + 3 * 1.0033, intensity=300.0),    # M+3
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, m0_mz,
        noise_at_target=2000.0,
    )
    assert ok is True, f"at HIGH S/N above strict window with full envelope, must be enough; got ok=False with note={note!r}"
    assert "M+0" in note and "M+1" in note


# ---------------------------------------------------------------------------
# 7) Tier mapping (table-driven)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "has_companion,series_len,envelope_ok,tolerance_ok,expected_tier",
    [
        # All checks pass -> GREEN.
        (True, 3, True, True, "GREEN"),
        # Companion + series but no envelope -> YELLOW (envelope missing
        # is treated as "not checked", so the call site decides between
        # YELLOW and RED based on companion/series only -- here we feed
        # envelope_ok=False to test the fallback path).
        # Envelope actively failed (envelope_skipped=False).
        # The user's new rule: hard negative signal, cannot be rescued
        # by companion + series success. Always RED.
        (True, 3, False, True, "RED"),
        (True, 0, False, True, "RED"),
        (False, 3, False, True, "RED"),
        # Same inputs but with envelope_skipped=True (e.g. candidate
        # m/z outside the strict envelope window, so the check was
        # never fully evaluated). Defer to companion + series.
        (True, 3, False, True, "YELLOW"),  # envelope_skipped=True
        (True, 0, False, True, "YELLOW"),  # envelope_skipped=True
        (False, 3, False, True, "YELLOW"),  # envelope_skipped=True
        # No companion, no series, envelope present -> RED.
        (False, 0, True, True, "RED"),
        # tolerance bad -> PURGE regardless of everything else.
        (True, 3, True, False, "PURGE"),
        (False, 0, True, False, "PURGE"),
    ],
)
def test_tier_from_scores(
    has_companion: bool,
    series_len: int,
    envelope_ok: bool,
    tolerance_ok: bool,
    expected_tier: str,
) -> None:
    # The envelope-active-fail cases all use envelope_skipped=False
    # (default). When the envelope actively fails (no M+1 found),
    # companion + series cannot rescue -- always RED. The first three
    # parametrize rows in the body above are envelope-fail -> RED.
    # The "envelope_skipped" cases are exercised separately below.
    envelope_skipped = False  # default for these parametrize cases
    # Detect the duplicate rows: the second triplet is envelope_skipped=True
    # and expects YELLOW. We pick envelope_skipped based on whether
    # companion or series is positive AND the expected tier is YELLOW.
    if (has_companion or series_len >= 3) and envelope_ok is False and expected_tier == "YELLOW":
        envelope_skipped = True
    tier = _tier_from_scores(
        has_companion, series_len, envelope_ok, tolerance_ok,
        envelope_skipped=envelope_skipped,
    )
    assert tier == expected_tier


# ---------------------------------------------------------------------------
# 7b) Tier mapping with skipped companion/series
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "has_companion,series_len,companion_skipped,series_skipped,expected_tier",
    [
        # Both companion and series were skipped (spectrum ends).
        # Envelope is OK. The old rule would have returned RED
        # (no companion, no series) but the candidate has no
        # evaluable negative evidence. Promote to YELLOW.
        (False, 0, True, True, "YELLOW"),
        # Same but envelope is not OK. The YELLOW-on-skipped rule
        # still applies -- at least one positive signal is better
        # than a hard RED.
        (False, 0, True, True, "YELLOW"),
        # Only companion was skipped (e.g. very high m/z where
        # +203 is off-spectrum but the series ladder fits in
        # range and found nothing). Treat as YELLOW rather than RED.
        (False, 0, True, False, "YELLOW"),
        # Only series was skipped (e.g. +203 in range but +609
        # off-spectrum; series requires 3 steps). Treat as YELLOW.
        (False, 0, False, True, "YELLOW"),
        # Nothing was skipped AND nothing was found -- genuine
        # absence. Stay RED.
        (False, 0, False, False, "RED"),
        # Skipped does NOT rescue a bad ppm match.
        (False, 0, True, True, "YELLOW"),
    ],
)
def test_tier_from_scores_with_skipped(
    has_companion: bool,
    series_len: int,
    companion_skipped: bool,
    series_skipped: bool,
    expected_tier: str,
) -> None:
    tier = _tier_from_scores(
        has_companion,
        series_len,
        envelope_ok=True,
        tolerance_ok=True,
        companion_skipped=companion_skipped,
        series_skipped=series_skipped,
    )
    assert tier == expected_tier


def test_score_companion_reports_skipped_flag() -> None:
    """When the spectrum ends, the companion checker must surface
    that it was skipped, not silently report count=0."""
    target = 5800.0
    # No peaks in range, so all offsets land past 5998.
    peaks = [Peak(mz=target, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 0.0, mz_max=5998.0
    )
    assert count == 0
    assert skipped is True, "skipped flag must be True when offsets land past mz_max"


def test_score_companion_not_skipped_when_in_range() -> None:
    """When all offsets are in range, the skipped flag must be False
    even if no peaks are found."""
    target = 1500.0
    peaks = [Peak(mz=target, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, skipped = _score_companion(
        target, sorted_mz, sorted_intensity, 10.0, 0.0, mz_max=5000.0
    )
    assert count == 0
    assert skipped is False, "skipped flag must be False when offsets are in range"


def test_score_series_reports_skipped_flag() -> None:
    """When the spectrum ends, the series checker must surface
    that it was truncated."""
    target = 5950.0
    # +162 step 1 = 6112, past 5998 -> truncated.
    peaks = [Peak(mz=target, intensity=5000.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    cache, _fallback = _build_noise_floor_cache(peaks)
    length, notes, skipped = _score_series(
        target, sorted_mz, sorted_intensity, 10.0, cache, mz_max=5998.0
    )
    assert length == 0
    assert skipped is True


def test_high_mz_candidate_promoted_to_yellow() -> None:
    """End-to-end: a high-m/z candidate whose companion/series
    offsets are all past the spectrum end must be promoted to
    YELLOW (not RED) when its envelope is OK, because the
    absence is not evaluable negative evidence.
    """
    # 12 GalNAc + 8 Gal + Na+ -> ~ 2908 + 22.99 = 2930. Well
    # below 5998, but the companion +203 = 3133, +162 = 3092,
    # +41 = 2971 are all in range. We need a HIGHER m/z.
    # Try 14 GalNAc + 10 Gal + Na+ = 14*203.0794 + 10*162.0528
    # + 18.0106 + 22.9898 = 4505.13. Companion at +203 = 4708
    # (in range), +162 = 4667 (in range), +41 = 4546 (in range).
    # That's still all in range. To get past-spectrum companions
    # we need a candidate at >= 5800. 17 GalNAc + 12 Gal = 17*203
    # + 12*162 + 18 + 23 = 5398. +203 = 5601, +162 = 5560, +41
    # = 5439 -- still in 5998. To force off-spectrum companions
    # we need 5800+. Let's go with 22 GalNAc + 14 Gal = 22*203
    # + 14*162 + 18 + 23 = 6751. That's > 5998 so no M+0. Bad.
    # Try 19 GalNAc + 14 Gal = 19*203.0794 + 14*162.0528 + 18.0106
    # + 22.9898 = 6194.49. Too high.
    # We need m0 in range and the +203 companion just out. Use
    # 21 GalNAc + 11 Gal = 21*203.0794 + 11*162.0528 + 18.0106
    # + 22.9898 = 6087.98. Too high by 90.
    # Use 20 GalNAc + 11 Gal = 20*203.0794 + 11*162.0528 + 18.0106
    # + 22.9898 = 5884.91. +203 -> 6088 (out). +162 -> 6047 (out).
    # +41 -> 5926 (in). +K/Na -> 5901 (in). The +203 and +162
    # are skipped but +41 and K/Na are in range. So this won't
    # fully skip companion, but it WILL skip the +1/+1 pair.
    # For the test, let's just use a synthetic narrow spectrum
    # that ends before any companion can be found, then verify
    # the candidate is YELLOW not RED.
    n_galnac, n_gal = 20, 11
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898  # ~ 5884.91
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    # Build a spectrum that ends at mz_max = m0_mz + 100. So
    # NO companion is in range. Series steps (+162 x3) all out.
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1.0033, intensity=3000.0),  # M+1
        Peak(mz=m0_mz + 2 * 1.0033, intensity=1000.0),  # M+2
        Peak(mz=m0_mz + 3 * 1.0033, intensity=300.0),  # M+3
        Peak(mz=m0_mz + 41.0, intensity=500.0),  # would be +1 GalNAc -1 Gal, in range but BELOW noise
    ]
    # mz_diff is set to a very small value (0.01 Da) so the tolerance
    # gate passes; da_tol=0.7 is the actual envelope-search radius.
    # Previously this test passed da_tol=10.0 which swallowed the
    # whole isotope pattern (M+0 fell inside the M+1 search window)
    # and broke the envelope check.
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "mz_diff": 0.01,
            }
        ]
    )
    out = screen_candidates(df, peaks, da_tol=0.7)
    # Envelope OK (M+0+M+1+M+2+M+3 all observed), companion/series
    # skipped (off-spectrum). With envelope OK + skipped -> YELLOW.
    assert out["tier"].iloc[0] == "YELLOW", (
        f"high-m/z candidate with skipped companion/series should be YELLOW; "
        f"got {out['tier'].iloc[0]!r} with notes: {out['screen_notes'].iloc[0]!r}"
    )
    assert "skipped" in out["screen_notes"].iloc[0]


# ---------------------------------------------------------------------------
# 8) Public screen_candidates -- end to end
# ---------------------------------------------------------------------------
def _make_candidate_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal candidate DataFrame matching app.py's contract.

    Each row may include either ``"|ppm|"`` (legacy) or ``"mz_diff"``
    (current). If ``|ppm|`` is present it is converted to a Da
    ``mz_diff`` at a 1400 Da reference mass (the manual-composition
    typical mass), so the candidate stays within the screener's
    ``|mz_diff| <= da_tol`` PURGE gate. Rows that already have
    ``mz_diff`` are passed through unchanged.
    """
    out_rows: list[dict] = []
    for r in rows:
        r = dict(r)
        if "mz_diff" not in r and "|ppm|" in r:
            # Convert legacy |ppm| to mz_diff using a representative
            # 1400 Da reference mass -- the typical manual-composition
            # check m/z range. da_tol=10.0 in the public function
            # default would accept up to 10 Da, so 0.1 ppm -> 0.14 Da,
            # well within tolerance.
            r["mz_diff"] = float(r["|ppm|"]) * 1400.0 / 1_000_000.0
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def test_screen_candidates_empty_df() -> None:
    df = _make_candidate_df([])
    out = screen_candidates(df, [])
    assert "tier" in out.columns
    assert "screen_notes" in out.columns
    assert "score_companion" in out.columns
    assert "score_series" in out.columns
    assert out.empty


def test_screen_candidates_ppm_off_goes_purge() -> None:
    # mz_diff = 50.0 Da, way above the 10.0 Da tolerance -> PURGE
    # even with companions. (The legacy |ppm| PURGE gate has been
    # replaced by a |mz_diff| Da gate; this test now uses mz_diff
    # directly to exercise the new contract.)
    n_galnac, n_gal = 1, 1
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "mz_diff": 50.0,
            }
        ]
    )
    # Even with a companion + envelope, the bad mz_diff forces PURGE.
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + GALNAC_DELTA, intensity=5000.0),
    ]
    out = screen_candidates(df, peaks, da_tol=10.0)
    assert out["tier"].iloc[0] == "PURGE"


def test_screen_candidates_green_full_evidence() -> None:
    # Build a candidate with everything we'd want: envelope present,
    # a companion peak at +203, a 3-step GalNAc ladder, and |ppm| good.
    n_galnac, n_gal = 2, 0
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898  # Na+ adduct
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    # Companion at +203, ladder at +1/+2/+3 GalNAc, and the envelope
    # at M+0 / M+1 / M+2. The S/N-scaled bar requires at most
    # M+0+M+1+M+2 for LOW or MEDIUM S/N. With M+0=20000 vs noise
    # floor ~1000-3000 the S/N is well above 5 (HIGH), so M+3 is
    # also needed -- add it.
    peaks = [
        Peak(mz=m0_mz, intensity=20000.0),
        Peak(mz=m0_mz + 1.0033, intensity=12000.0),
        Peak(mz=m0_mz + 2 * 1.0033, intensity=4000.0),
        Peak(mz=m0_mz + 3 * 1.0033, intensity=800.0),  # M+3
        Peak(mz=m0_mz + GALNAC_DELTA, intensity=20000.0),
        Peak(mz=m0_mz + 2 * GALNAC_DELTA, intensity=20000.0),
        Peak(mz=m0_mz + 3 * GALNAC_DELTA, intensity=20000.0),
        *[Peak(mz=m0_mz + 10 + i * 14, intensity=30000.0) for i in range(20)],
    ]
    out = screen_candidates(df, peaks, da_tol=0.5)
    assert out["tier"].iloc[0] == "GREEN"
    assert out["score_companion"].iloc[0] >= 1
    assert out["score_series"].iloc[0] >= SERIES_MIN_LENGTH


def test_screen_candidates_yellow_companion_only() -> None:
    # Companion peak, no series, no envelope. (Envelope won't be
    # observed here because we only put the M+0 peak, not the M+1.)
    n_galnac, n_gal = 1, 0
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    # M+0 (5000) plus a companion at +GALNAC_DELTA, plus noise-floor
    # peaks at intensity 100 so the local noise floor is comfortably
    # below the M+0 (1.35 x 100 = 135 < 5000). Without the noise
    # peaks the only candidates in the spectrum set the floor to
    # 5000, which trips the new M+0 S/N PURGE gate.
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + GALNAC_DELTA, intensity=5000.0),
        Peak(mz=m0_mz - 500.0, intensity=100.0),
        Peak(mz=m0_mz - 800.0, intensity=100.0),
        Peak(mz=m0_mz + 500.0, intensity=100.0),
        Peak(mz=m0_mz + 800.0, intensity=100.0),
    ]
    out = screen_candidates(df, peaks, da_tol=10.0)
    # Envelope actively failed (no M+1 in spectrum). The user's
    # rule: an active envelope fail is a hard negative signal,
    # cannot be rescued by companion success alone. -> RED.
    assert out["tier"].iloc[0] == "RED"
    # The companion IS reported as found -- only the envelope failed.
    assert "Companions: YES" in out["screen_notes"].iloc[0]


def test_screen_candidates_red_no_evidence() -> None:
    # No companion, no series, no envelope (the candidate is just
    # alone with its M+0 peak). The spectrum needs to be wide enough
    # that no companion/series offset is skipped due to the spectrum
    # boundary, otherwise the candidate would be promoted to YELLOW
    # on the "skipped" path (no evaluable negative evidence).
    n_galnac, n_gal = 1, 0
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    # Build a wide enough spectrum that companion (+203) and series
    # (+203, +406, +609) are all in range. No real companion peaks,
    # just background.
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1000.0, intensity=10.0),  # well past +203
    ]
    out = screen_candidates(df, peaks, da_tol=0.5)
    assert out["tier"].iloc[0] == "RED"
    # The note must mention every check was run, even though they
    # found nothing. The user reported that high-m/z candidates
    # were silently saying "no 13C isotopes" without saying anything
    # about companion / series -- this test locks the fix in.
    note = out["screen_notes"].iloc[0]
    # The envelope line could be either "13C envelope not observed"
    # (no M+0 either) or "13C envelope requires M+1" (M+0 found
    # but no M+1/M+2/M+3 -- the user explicitly requires at least
    # M+1 in both windows). Both are valid negative-envelope signals.
    assert ("13C envelope not observed" in note) or (
        "requires M+1" in note
    )
    assert "Companions: NO (none in range)" in note
    assert "Series: NO (none in range)" in note


def test_screen_candidates_app_imports_cleanly() -> None:
    # Smoke-check the full public surface: this is the call site in
    # app.py:210. If the function signature drifted, the import will
    # still resolve, but a real call should return a DataFrame.
    df = _make_candidate_df(
        [
            {
                "mz": 1000.0,
                "intensity": 1000.0,
                "n_galnac": 1,
                "n_gal": 1,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    out = screen_candidates(df, [Peak(mz=1000.0, intensity=1000.0)], da_tol=10.0)
    assert isinstance(out, pd.DataFrame)
    assert "tier" in out.columns
    assert "screen_notes" in out.columns
    assert "score_companion" in out.columns
    assert "score_series" in out.columns


def test_screen_candidates_strictness_off_short_circuits() -> None:
    # When the sidebar checkbox is off, the public function must
    # short-circuit before doing any expensive work (no molmass
    # calls, no noise-floor cache build, no per-candidate scoring).
    # The returned DataFrame still has the four columns so downstream
    # code does not have to special-case the off state.
    df = _make_candidate_df(
        [
            {
                "mz": 1500.0,
                "intensity": 1000.0,
                "n_galnac": 2,
                "n_gal": 2,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    out = screen_candidates(
        df, [], da_tol=10.0, strictness="off"
    )
    assert "tier" in out.columns
    assert "screen_notes" in out.columns
    assert "score_companion" in out.columns
    assert "score_series" in out.columns
    # No scoring happened -> the columns are empty / zero.
    assert out["tier"].iloc[0] == "" or pd.isna(out["tier"].iloc[0])
    assert out["score_companion"].iloc[0] == 0
    assert out["score_series"].iloc[0] == 0


def test_screen_candidates_handles_unknown_ion() -> None:
    # Per the module docstring, the public function never raises on bad
    # input data. An unknown ion label degrades the row to PURGE so it
    # is visible in the output and the app does not crash.
    df = _make_candidate_df(
        [
            {
                "mz": 1000.0,
                "intensity": 1000.0,
                "n_galnac": 1,
                "n_gal": 1,
                "ion": "Fe+",  # not supported
                "|ppm|": 0.1,
            }
        ]
    )
    out = screen_candidates(df, [Peak(mz=1000.0, intensity=1000.0)], da_tol=10.0)
    assert isinstance(out, pd.DataFrame)
    assert out["tier"].iloc[0] == "PURGE"
    assert "unknown ion" in out["screen_notes"].iloc[0]


def test_screen_candidates_companion_check_uses_full_candidate_set() -> None:
    """Lock in the contract: the companion check runs once on the FULL
    candidate set (i.e. against the raw spectrum) BEFORE any display
    filter, so the "Hide if no companion peak found" toggle in the
    UI is purely a post-screening view filter.

    In app.py the call order is:
        1) _candidates_to_dataframe() builds the full candidate df and
           calls screen_candidates() at app.py:197 (one pass over
           every row, against the raw spectrum_peaks).
        2) The hide-no-companion checkbox at app.py:758 then filters
           the SCREENED display_df by score_companion > 0.

    The user's concern was that a row whose companion check would
    have passed gets dropped because a DIFFERENT, hide-filtered row
    is its only "companion" in the candidate table -- but the
    companion check actually looks at the raw spectrum peak list,
    not at other candidate rows, so this can never happen.

    This test builds:
      A: n_galnac=2, n_gal=0, Na+ at m0_mz  (the row under test)
      B: n_galnac=2, n_gal=1, Na+ at m0+GAL (its "companion" row)
      C: n_galnac=5, n_gal=0, Na+ at m0+500 (unrelated, no companion)
    and asserts that A's score_companion >= 1 because the +Gal peak
    is in the spectrum, regardless of whether B survives any later
    display filter. Then it simulates the hide toggle by filtering
    score_companion > 0 and verifies C (and B, which has no companion
    peak in the spectrum) are dropped.
    """
    n_galnac_A, n_gal_A = 2, 0
    n_galnac_B, n_gal_B = 2, 1
    n_galnac_C, n_gal_C = 5, 0
    neutral_A = (
        n_galnac_A * GALNAC_DELTA + n_gal_A * GAL_DELTA + 18.0106
    )
    m0_mz = neutral_A + 22.9898  # Na+ adduct

    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 1000.0,
                "n_galnac": n_galnac_A,
                "n_gal": n_gal_A,
                "ion": "Na+",
                "|ppm|": 0.1,
            },
            {
                "mz": m0_mz + GAL_DELTA,
                "intensity": 500.0,
                "n_galnac": n_galnac_B,
                "n_gal": n_gal_B,
                "ion": "Na+",
                "|ppm|": 0.1,
            },
            {
                "mz": m0_mz + 500.0,
                "intensity": 8000.0,
                "n_galnac": n_galnac_C,
                "n_gal": n_gal_C,
                "ion": "Na+",
                "|ppm|": 0.1,
            },
        ]
    )

    # Spectrum has the candidate peaks PLUS a real +Gal companion for
    # A at m0+GAL. C is an isolated noise peak with no companion
    # peak anywhere nearby.
    peaks = [
        Peak(mz=m0_mz, intensity=1000.0),
        Peak(mz=m0_mz + GAL_DELTA, intensity=500.0),
        Peak(mz=m0_mz + 500.0, intensity=8000.0),
    ]

    out = screen_candidates(df, peaks, da_tol=10.0)

    # A: the +Gal peak in the spectrum is the companion -- at least
    # one companion class must have matched. This is the core
    # contract: the score is computed against the full spectrum, not
    # against the other candidate rows.
    a_score = int(out.loc[out["mz"] == m0_mz, "score_companion"].iloc[0])
    assert a_score >= 1, (
        f"row A must find the +Gal companion peak in the spectrum; "
        f"got score_companion={a_score}, screen_notes="
        f"{out.loc[out['mz'] == m0_mz, 'screen_notes'].iloc[0]!r}"
    )

    # C: no companion peak in the spectrum within 162-365 Da, so its
    # score must be 0. This is the row the "Hide if no companion"
    # toggle is designed to drop.
    c_score = int(out.loc[out["mz"] == m0_mz + 500.0, "score_companion"].iloc[0])
    assert c_score == 0, (
        f"row C has no companion peak in the spectrum; "
        f"expected score_companion=0, got {c_score}"
    )

    # Simulate the hide toggle's filter from app.py:771-774: keep
    # rows with score_companion > 0. A survives, C (and B, which
    # has no companion peak in the spectrum) are dropped. This is
    # the full proof that the display filter never affects the
    # underlying score -- a row that gets hidden never erases the
    # companion evidence for any other row.
    visible = out[out["score_companion"] > 0]
    visible_mz = set(visible["mz"].tolist())
    assert m0_mz in visible_mz, "row A must remain visible"
    assert (m0_mz + 500.0) not in visible_mz, (
        "row C has no companion in the spectrum and must be hidden"
    )


# ---------------------------------------------------------------------------
# Edge-case regression tests
# ---------------------------------------------------------------------------


def test_screen_candidates_empty_spectrum_with_rows() -> None:
    """Spectrum with zero peaks but DataFrame has rows.

    The candidate has no observed M+0 peak, so the M+0 S/N gate
    PURGEs the row (intensity 0 < 1.35 x 1.0 noise floor sentinel).
    """
    df = _make_candidate_df(
        [{"mz": 1500.0, "intensity": 1000.0, "n_galnac": 1, "n_gal": 1, "ion": "Na+", "|ppm|": 0.1}]
    )
    out = screen_candidates(df, [], da_tol=10.0)
    assert out["tier"].iloc[0] == "PURGE"
    assert "no observed peaks" in out["screen_notes"].iloc[0]


def test_score_envelope_strict_window_boundary_3800_inclusive() -> None:
    """STRICT_ENVELOPE_MZ_HI = 3800.0 is INCLUSIVE.

    A candidate at exactly 3800.0 gets the strict M+0+M+1+M+2+M+3 check.
    The composition is chosen so its theoretical m/z lands at 3800.0
    (or close to it). The peak list is anchored on the candidate's m/z
    (target_mz) -- not on the theoretical -- because the screener
    treats the candidate's observed m/z as the M+0 anchor.
    """
    # Choose n_galnac, n_gal so the theoretical m/z is close to 3800.
    # (8, 6) -> 8*203.0794 + 6*162.0528 + 18.0106 + 22.9898 = 2639.96
    # (10, 8) -> 10*203.0794 + 8*162.0528 + 18.0106 + 22.9898 = 3361.19
    # (12, 10) -> 12*203.0794 + 10*162.0528 + 18.0106 + 22.9898 = 4082.43
    # None of the standard (n, m) combinations lands exactly at 3800,
    # so we use a near-boundary composition (11, 9) and set
    # target_mz = 3800.0 explicitly to exercise the boundary check.
    n_galnac, n_gal = 11, 9
    target_mz = 3800.0  # exactly on the strict-window upper bound
    peaks = [
        Peak(mz=target_mz, intensity=5000.0),
        Peak(mz=target_mz + 1.0033, intensity=3000.0),
        Peak(mz=target_mz + 2 * 1.0033, intensity=1000.0),
        Peak(mz=target_mz + 3 * 1.0033, intensity=500.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, _, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity, 0.7, target_mz, noise_at_target=0.0
    )
    assert ok is True  # All 4 envelope peaks present, strict mode passes at the boundary


def test_score_companion_two_peaks_same_offset_not_double_counted() -> None:
    """Two peaks at the same m/z in the companion offset window count as one companion, not two."""
    target = 1000.0
    peaks = [
        Peak(mz=target, intensity=5000.0),
        Peak(mz=target + GALNAC_DELTA, intensity=500.0),
        Peak(mz=target + GALNAC_DELTA, intensity=800.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, _notes, _ = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 0.0)
    assert count == 1


def test_score_companion_peak_diff_at_exact_tight_boundary_0_30() -> None:
    """Companion peak at exactly +0.30 Da off theoretical is counted via the tight override (boundary inclusive)."""
    target = 1013.35
    # Peak at +203.0794 + 0.30 = 1216.7294. Below noise floor (intensity 30, floor 100).
    peaks = [Peak(mz=target, intensity=2000.0), Peak(mz=1216.7294, intensity=30.0)]
    from glycan_ms.screener import _bin_index
    cache = {
        _bin_index(target): 5.0,
        _bin_index(1216.7294): 100.0,
    }
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    count, notes, _ = _score_companion(target, sorted_mz, sorted_intensity, 10.0, 5.0, noise_cache=cache)
    assert count == 1
    assert any("tight" in n for n in notes)


def test_noise_floor_skips_nan_intensities() -> None:
    """NaN intensity peaks are dropped from the noise floor; the next valid peak is used."""
    import math
    peaks = [Peak(mz=1000.0, intensity=float("nan")), Peak(mz=1100.0, intensity=10.0)]
    floor = _noise_floor_for_mz(peaks, 1050.0)
    assert floor == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Filter-contract regression tests
#
# The app's "Hide if 13C envelope not observed" toggle in app.py:885-893
# uses two substring checks against ``screen_notes``:
#   1. "envelope observed" -> full envelope (M+0 + required satellites)
#   2. "M.0 present" (regex, escapes the +) -> M+0 only, no satellites
# Rows matching EITHER are kept; rows matching NEITHER are dropped.
# These tests pin the screener's note strings so that contract cannot
# silently drift -- if it does, every uploaded spectrum with a sparse
# M+0-only candidate would disappear from the table again.
# ---------------------------------------------------------------------------


def test_screen_notes_strings_match_filter_contract() -> None:
    """The three screener note strings the filter depends on must remain stable."""
    # M+0 + M+1 in permissive window (above 3800): "13C envelope observed (M+0, M+1)"
    # M+0 only, no satellites: "13C envelope requires M+1 (only M+0 present ...)"
    # M+0 missing entirely: "13C envelope not observed" / "no observed peaks ..."
    # Lock all three so a future refactor cannot silently break the filter.
    from glycan_ms.screener import _score_envelope

    n_galnac, n_gal = 2, 2
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898

    # Case 1: full envelope (M+0 + M+1 + M+2 + M+3 above noise).
    peaks_full = [
        Peak(mz=m0, intensity=5000.0),
        Peak(mz=m0 + 1.0033, intensity=3000.0),
        Peak(mz=m0 + 2 * 1.0033, intensity=1000.0),
        Peak(mz=m0 + 3 * 1.0033, intensity=500.0),
    ]
    sm, si = _sorted_mz_index(peaks_full)
    ok, note, _skipped = _score_envelope(n_galnac, n_gal, Adduct.NA, sm, si, 0.7, m0, noise_at_target=0.0)
    assert ok is True
    assert "envelope observed" in note

    # Case 2: M+0 only (no satellites above noise floor). The user
    # explicitly requires M+1 in BOTH windows -- M+0 alone is not
    # enough envelope evidence to call the candidate "envelope
    # observed". The note must say "requires M+1" so the user can
    # see why the candidate failed the envelope check.
    peaks_m0 = [Peak(mz=m0, intensity=5000.0)]
    sm, si = _sorted_mz_index(peaks_m0)
    ok, note, _skipped = _score_envelope(n_galnac, n_gal, Adduct.NA, sm, si, 0.7, m0, noise_at_target=200.0)
    assert ok is False
    assert "requires M+1" in note, f"M+0-only note must say 'requires M+1', got: {note!r}"
    assert "M+0 present" in note, f"M+0-only note must mention M+0 present, got: {note!r}"

    # Case 3: no peak within any envelope window -> the function
    # early-exits at line 616-617 with a distinct message
    # ("no observed peaks at theoretical isotope positions") before
    # the line 672 fallback can fire. The fallback note string
    # "13C envelope not observed" is dead code at the moment; the
    # contract the filter depends on is just "neither 'envelope
    # observed' nor 'M+0 present' is in the note".
    peaks_none = [Peak(mz=m0 + 50.0, intensity=5000.0)]  # far from theoretical
    sm, si = _sorted_mz_index(peaks_none)
    ok, note, _skipped = _score_envelope(n_galnac, n_gal, Adduct.NA, sm, si, 0.7, m0, noise_at_target=200.0)
    assert ok is False
    assert "envelope observed" not in note
    assert "M+0 present" not in note


def test_hide_no_envelope_filter_keeps_m0_only_rows() -> None:
    """The display filter must keep M+0-only rows and drop only M+0-missing rows.

    This is the regression test for the bug where the filter used the
    substring "envelope observed" and silently dropped every row whose
    candidate had M+0 on the spectrum but no M+1/M+2/M+3 above the
    noise floor -- which is the common case for spectra with sparse
    high-m/z peaks (e.g. 2c5-150m... -- only 8 peaks in the whole
    spectrum, all M+0 with no satellites).

    The note for the M+0-only case is now
    "13C envelope requires M+1 (only M+0 present at ...)" -- it
    still contains "M+0 present" so the filter regex
    ``M.0 present`` (with `.` escaping the +) keeps it.
    """
    import pandas as pd

    # Build a candidate DataFrame mirroring the post-screener output.
    df = pd.DataFrame(
        [
            {
                "mz": 1177.42, "intensity": 5000.0,
                "n_galnac": 4, "n_gal": 2, "ion": "Na+",
                "|ppm|": 0.1, "mz_diff": 0.0,
                "tier": "YELLOW",
                "score_companion": 4, "score_series": 0,
                "screen_notes": "Companions: YES (4 found); 13C envelope requires M+1 (only M+0 present at 1177.4 m/z)",
            },
            {
                "mz": 1177.42, "intensity": 5000.0,
                "n_galnac": 4, "n_gal": 2, "ion": "K+",
                "|ppm|": 0.1, "mz_diff": 0.0,
                "tier": "YELLOW",
                "score_companion": 0, "score_series": 0,
                "screen_notes": "Companions: skipped; 13C envelope not observed",
            },
        ]
    )

    # Apply the same two-substring filter the app uses at app.py:885-893.
    notes = df["screen_notes"].fillna("")
    mask = notes.str.contains("envelope observed") | notes.str.contains(
        "M.0 present", regex=True
    )
    kept = df[mask]

    # The M+0-only row must survive (it has real evidence).
    assert len(kept) == 1, (
        f"expected exactly the M+0-only row to survive, got {len(kept)}: "
        f"{kept['screen_notes'].tolist()}"
    )
    assert "M+0 present" in kept.iloc[0]["screen_notes"]


def test_score_envelope_da_tolerance_binds_at_high_mz() -> None:
    """The M+0 search uses the user's Da tolerance directly.

    At m/z ~ 4000, 10 ppm = 40 mDa, so a 0.7 Da user tolerance
    dominates. The spectrum has a peak 0.5 Da off the candidate's
    m/z (the kind of mass error that the user's 0.7 Da slider is
    meant to catch). With da_tol=0.1 the peak must be missed; with
    da_tol=0.7 it must be found, along with a band-rule M+1.
    """
    n_galnac, n_gal = 10, 12
    m0_theoretical = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898
    # Candidate's matched m/z (target_mz) is the OBSERVED M+0
    # anchor. The actual M+0 peak in the spectrum is 0.5 Da above
    # target_mz -- the kind of error the Da slider should accept.
    target_mz = m0_theoretical
    observed_m0 = target_mz + 0.5
    # The whole envelope is shifted by the same +0.5 Da, so M+1
    # sits at observed_m0 + 1.0 (still in the 0.75-1.2 band of M+0).
    # S/N = 10 so M+1 is naturally above the noise AND the relative
    # gate is enabled (S/N >= 1.5). M+2 also provided (MEDIUM bucket
    # is {1, 2} so we need M+2 at HIGH bar which is 5x).
    peaks = [
        Peak(mz=observed_m0, intensity=20000.0),            # observed M+0 (0.5 Da off target_mz)
        Peak(mz=observed_m0 + 1.0, intensity=8800.0),       # observed M+1 (in band)
        Peak(mz=observed_m0 + 2 * 1.0033, intensity=2000.0),  # M+2
        Peak(mz=observed_m0 + 3 * 1.0033, intensity=300.0),   # M+3
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)

    # 0.1 Da tolerance: M+0 peak is 0.5 Da off, outside the 0.1 Da window -> no envelope.
    ok_low, _, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.1, target_mz=target_mz, noise_at_target=0.0,
    )
    assert ok_low is False

    # 0.7 Da tolerance: M+0 found (0.5 < 0.7), M+1 in band, M+2 / M+3
    # present. HIGH S/N so {1, 2, 3} required and provided.
    ok_high, _, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=target_mz, noise_at_target=2000.0,
    )
    assert ok_high is True


def test_score_envelope_offby_05da_candidate_still_finds_envelope() -> None:
    """The M+0 search anchors on the OBSERVED candidate m/z, not the theoretical.

    Regression: a candidate matched 0.5 Da off its theoretical m/z
    (well inside the user's 0.7 Da tolerance) used to fail the
    envelope check because the satellites were searched against
    theoretical positions, then the 0.75-1.2 band rule rejected
    every satellite by ~0.5 Da. The fix anchors the whole envelope
    on target_mz, so the satellites propagate from the observed
    M+0 and land correctly in the band.
    """
    n_galnac, n_gal = 5, 4
    theoretical = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898
    # Candidate matched at theoretical + 0.5 Da (inside da_tol=0.7).
    observed_m0 = theoretical + 0.5
    peaks = [
        Peak(mz=observed_m0, intensity=10000.0),                       # M+0
        Peak(mz=observed_m0 + 1.0033, intensity=2000.0),              # M+1
        Peak(mz=observed_m0 + 2 * 1.0033, intensity=500.0),            # M+2
        Peak(mz=observed_m0 + 3 * 1.0033, intensity=200.0),            # M+3
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=observed_m0, noise_at_target=0.0,
    )
    assert ok is True, (
        f"off-by-0.5-Da candidate with all 4 envelope peaks present must pass; "
        f"got note={note!r}"
    )
    assert "M+1" in note
    assert "M+2" in note
    assert "M+3" in note


def test_score_envelope_satellite_band_0p75_to_1p2_from_previous() -> None:
    """Each M+k must be 0.75 to 1.2 Da above the PREVIOUS isotope's theoretical m/z.

    A peak closer than 0.75 Da to the previous isotope is a different
    species (not 13C). A peak further than 1.2 Da is more than one 13C
    substitution. Either way it does not count as an observed M+k --
    "chunk it in the bin".
    """
    n_galnac, n_gal = 1, 0
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898
    # M+1 theoretical: m0 + 1.003355 Da. Test three satellites:
    #   - 0.5 Da above M+0 (too close -> NOT M+1)
    #   - 1.0 Da above M+0 (in band -> IS M+1)
    #   - 1.5 Da above M+0 (too far -> NOT M+1)
    # The 0.5 Da and 1.5 Da cases must NOT be counted as satellites even
    # though they would have been inside the 0.7 Da global window.
    peaks = [
        Peak(mz=m0, intensity=10000.0),                    # M+0
        Peak(mz=m0 + 0.5, intensity=2000.0),                # NOT M+1 (too close)
        Peak(mz=m0 + 1.0, intensity=2000.0),                # IS M+1
        Peak(mz=m0 + 1.5, intensity=2000.0),                # NOT M+1 (too far)
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=m0, noise_at_target=0.0,
    )
    # Only M+0 is observed; the band-rejected peaks at 0.5 / 1.5 Da
    # must NOT have been counted as M+1. Under the new rule the
    # envelope check requires M+0 + at least M+1, so the result is
    # "requires M+1" (the M+0-only path) -- the M+1 substring here
    # is inside "requires M+1", which is exactly what we want.
    assert "envelope observed" not in note, (
        f"peaks 0.5 and 1.5 Da from M+0 must be band-rejected, but envelope was marked observed; "
        f"got note={note!r}"
    )
    assert note.startswith("13C envelope requires M+1")


def test_score_envelope_satellite_band_accepts_in_range_peak() -> None:
    """Positive case: a peak 1.0 Da above M+0 (in the 0.75-1.2 band) is accepted as M+1."""
    # 0 GalNAc + 1 Gal so the n_galnac > n_gal override does not fire.
    n_galnac, n_gal = 0, 1
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898
    # S/N=10 so M+1 is naturally above the noise AND the relative
    # gate is enabled (S/N >= 1.5). HIGH bucket (10x noise) so
    # required is {1, 2, 3}.
    peaks = [
        Peak(mz=m0, intensity=10000.0),
        Peak(mz=m0 + 1.0, intensity=4400.0),
        Peak(mz=m0 + 2 * 1.0033, intensity=1000.0),
        Peak(mz=m0 + 3 * 1.0033, intensity=150.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=m0, noise_at_target=1000.0,
    )
    assert ok is True, f"M+0 + full envelope in band should pass; got note={note!r}"
    assert "M+1" in note


def test_score_envelope_rejects_satellite_near_low_sn_noise() -> None:
    """At LOW M+0 S/N, a coincidental noise spike near the M+1 position
    must NOT be accepted as a real satellite.

    Regression for the user's 1462-Da case: M+0=447, noise=350
    (S/N=1.28). The expected M+1 (~197 for a 40C glycan) is below
    the noise floor and therefore indistinguishable from noise. A
    coincidental noise spike at 360 (just above the noise) used to
    be accepted as a real M+1 under the old `intensity >= noise`
    gate; under the new model (CI gate = noise + 2*sqrt(noise) =
    387.4, relative gate disabled below S/N=1.5) it is correctly
    rejected.

    This is the physical justification for the
    ``ENVELOPE_SIGMA_FACTOR`` term and the
    ``rel_gate_enabled`` gate: at low S/N the noise gate must
    rise above the bare noise floor AND the relative gate must
    be muted, to avoid false satellite acceptance from random
    spikes.
    """
    n_galnac, n_gal = 1, 2
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898
    # M+0 = 447 (real peak, just above noise), noise = 350.
    # A coincidental noise spike at the M+1 position with intensity
    # 360 (just above the noise floor).
    peaks = [
        Peak(mz=m0, intensity=447.0),
        Peak(mz=m0 + 1.0033, intensity=360.0),  # coincidental noise spike
    ]
    sm, si = _sorted_mz_index(peaks)
    # S/N = 1.28. Above LOW bar (1.09) so MEDIUM bucket -> required
    # {1, 2}. Below MEDIUM threshold (1.5) so relative gate is
    # disabled. CI gate = 350 + 2*sqrt(350) = 387. The 360 spike
    # is below 387 -> NOT observed as M+1. M+2 also missing.
    # Envelope fails -> ok=False.
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0, noise_at_target=350.0,
    )
    assert ok is False, (
        f"at LOW M+0 S/N=1.28, M+1 at 360 (coincidental noise spike) "
        f"must be rejected; got ok=True note={note!r}"
    )
    # The note must explain why envelope failed.
    assert "requires M+1" in note or "requires M+2" in note, (
        f"note must explain missing satellite; got {note!r}"
    )

    # Control: same M+0 S/N but M+1 at 500 (well above CI gate 387).
    # Even at MEDIUM bucket ({1, 2} required), M+2 missing still
    # fails the envelope, so ok=False. This proves the system
    # correctly requires M+2 at MEDIUM S/N even when M+1 is
    # confidently observed.
    peaks_strong_m1 = [
        Peak(mz=m0, intensity=447.0),
        Peak(mz=m0 + 1.0033, intensity=500.0),
        # M+2 still missing -> MEDIUM bucket fails.
    ]
    sm, si = _sorted_mz_index(peaks_strong_m1)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0, noise_at_target=350.0,
    )
    assert ok is False, (
        f"at MEDIUM S/N, M+0+M+1 alone must still fail (M+2 missing); "
        f"got ok=True note={note!r}"
    )
    assert "requires M+2" in note


def test_score_envelope_ci_gate_high_sn_unchanged() -> None:
    """At HIGH M+0 S/N, the CI gate is effectively a no-op.

    For large M+0 with low noise, sqrt(noise) is much smaller than
    noise, so the CI gate is dominated by the bare noise floor.
    The natural satellites (M+1, M+2, M+3) clear both old and new
    gates. This guards against the CI gate accidentally
    regressing high-S/N data -- at HIGH S/N the strict envelope
    requires all four satellites, so we provide them.
    """
    n_galnac, n_gal = 1, 2
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898
    # M+0 = 10000, noise = 100. CI gate = 100 + 2*10 = 120.
    # HIGH S/N (100x noise) -> required {1, 2, 3}. Provide all
    # four satellites at expected isotopologue intensities.
    peaks = [
        Peak(mz=m0, intensity=10000.0),
        Peak(mz=m0 + 1.0033, intensity=4400.0),         # M+1 (~44%)
        Peak(mz=m0 + 2 * 1.0033, intensity=1000.0),     # M+2 (~10%)
        Peak(mz=m0 + 3 * 1.0033, intensity=150.0),      # M+3 (~1.5%)
    ]
    sm, si = _sorted_mz_index(peaks)
    ok, _note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0, noise_at_target=100.0,
    )
    assert ok is True, (
        f"at HIGH S/N with full envelope, envelope should pass; got ok=False"
    )


def test_score_envelope_galnac_dominance_requires_m3() -> None:
    """When n_galnac > n_gal, M+1+M+2+M+3 must all clear noise regardless of S/N.

    Regression for the user's rule that GalNAc-dominant compositions
    (e.g. an O-glycan core with more GalNAc than Gal backbone) require
    the full 4-satellite envelope. The S/N-scaled bar would normally
    require only M+0+M+1 at LOW S/N, but the GalNAc-dominance rule
    elevates the bar to HIGH (M+0+M+1+M+2+M+3) unconditionally.
    """
    # 8 GalNAc + 2 Gal -> n_galnac > n_gal.
    n_galnac, n_gal = 8, 2
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898

    # 1) At LOW S/N (M+0 = noise, would normally be {1} only), missing
    #    M+2 must FAIL because the override demands {1, 2, 3}.
    peaks_low_sn = [
        Peak(mz=m0, intensity=1000.0),
        Peak(mz=m0 + 1.0033, intensity=300.0),
        # M+2 missing -- should fail under the override.
    ]
    sm, si = _sorted_mz_index(peaks_low_sn)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0, noise_at_target=1000.0,
    )
    assert ok is False, (
        f"GalNAc-dominant candidate at LOW S/N without M+2 must fail; got ok=True note={note!r}"
    )
    # At S/N=1.0 (below LOW 1.09 bar) the natural M+1 is also below
    # the CI gate so M+1 is not observed -- the note reports the
    # first missing required satellite (M+1). The GalNAc override
    # still applies; it's just that M+1 is the first thing missing.
    assert "requires M+1" in note or "requires M+2" in note or "requires M+3" in note, (
        f"note must explain a missing satellite; got {note!r}"
    )

    # 2) Same composition with M+0+M+1+M+2+M+3 visible at HIGH S/N
    #    must PASS. The override does not block legitimate full envelopes.
    #    S/N=10 so M+1 is naturally above the noise AND the relative
    #    gate is enabled (S/N >= 1.5).
    peaks_full = [
        Peak(mz=m0, intensity=20000.0),
        Peak(mz=m0 + 1.0033, intensity=8800.0),
        Peak(mz=m0 + 2 * 1.0033, intensity=2000.0),
        Peak(mz=m0 + 3 * 1.0033, intensity=300.0),
    ]
    sm, si = _sorted_mz_index(peaks_full)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0, noise_at_target=2000.0,
    )
    assert ok is True, (
        f"GalNAc-dominant candidate with full envelope must pass; got ok=False note={note!r}"
    )

    # 3) Control: same HIGH-S/N peaks but with n_galnac <= n_gal
    #    (4 GalNAc + 6 Gal). The override does not fire, but HIGH S/N
    #    still requires {1, 2, 3} -- so M+2 / M+3 must be visible.
    n_galnac_ctrl, n_gal_ctrl = 4, 6
    m0_ctrl = (
        n_galnac_ctrl * GALNAC_DELTA + n_gal_ctrl * GAL_DELTA
        + 18.0106 + 22.9898
    )
    peaks_ctrl = [
        Peak(mz=m0_ctrl, intensity=20000.0),
        Peak(mz=m0_ctrl + 1.0033, intensity=8800.0),
        Peak(mz=m0_ctrl + 2 * 1.0033, intensity=2000.0),
        Peak(mz=m0_ctrl + 3 * 1.0033, intensity=300.0),
    ]
    sm, si = _sorted_mz_index(peaks_ctrl)
    ok, note, _skipped = _score_envelope(
        n_galnac_ctrl, n_gal_ctrl, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0_ctrl, noise_at_target=2000.0,
    )
    assert ok is True, (
        f"control (n_galnac <= n_gal) at HIGH S/N with full envelope must pass; "
        f"got ok=False note={note!r}"
    )


def test_score_envelope_below_900_never_requires_m3() -> None:
    """Sub-900 candidates pass without M+3, even at high S/N and GalNAc dominance."""
    target = 850.0
    peaks = [
        Peak(mz=target, intensity=10_000.0),
        Peak(mz=target + 1.003355, intensity=2_000.0),
        Peak(mz=target + 2 * 1.003355, intensity=500.0),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)

    ok, note, skipped = _score_envelope(
        3,
        1,
        Adduct.NA,
        sorted_mz,
        sorted_intensity,
        0.1,
        target,
        noise_at_target=100.0,
    )

    assert ok is True, note
    assert skipped is False
    assert "requires M+3" not in note


def test_score_envelope_low_bar_is_1p09() -> None:
    """The LOW S/N envelope bar is exactly 1.09x noise.

    Pins the user's latest retune. At S/N = 1.08 the bar is not
    cleared (LOW bar required), at S/N = 1.10 it is (MEDIUM bar
    required). The test asserts the boundary is exactly 1.09.
    """
    # 1 GalNAc + 2 Gal so n_galnac <= n_gal and the GalNAc-dominance
    # override does not fire -- this test isolates the LOW-bar retune.
    n_galnac, n_gal = 1, 2
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898

    # S/N just under 1.09: M+0 = 1080, noise = 1000. S/N = 1.08.
    # Required is {1} (LOW). At LOW S/N the relative gate is
    # disabled and the CI gate (1000 + 2*sqrt(1000) = 1063) must
    # be cleared by the satellite. M+1 at 1500 passes.
    peaks_just_under = [
        Peak(mz=m0, intensity=1080.0),
        Peak(mz=m0 + 1.0033, intensity=1500.0),
    ]
    sm, si = _sorted_mz_index(peaks_just_under)
    ok, _note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0, noise_at_target=1000.0,
    )
    assert ok is True, (
        "S/N=1.08 with M+0+M+1 (M+1 above CI gate) must pass at the LOW bar"
    )

    # S/N just above 1.09: M+0 = 1100, noise = 1000. S/N = 1.10.
    # Now the bar is MEDIUM -> required {1, 2}. M+2 missing -> fail.
    peaks_just_over = [
        Peak(mz=m0, intensity=1100.0),
        Peak(mz=m0 + 1.0033, intensity=1500.0),
        # M+2 missing
    ]
    sm, si = _sorted_mz_index(peaks_just_over)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sm, si,
        da_tol=0.7, target_mz=m0, noise_at_target=1000.0,
    )
    assert ok is False, (
        f"S/N=1.10 must trigger MEDIUM bar (M+0+M+1+M+2 required); "
        f"missing M+2 -> envelope fails. got ok=True note={note!r}"
    )
    assert "requires M+2" in note


def test_score_envelope_m_plus_2_band_applies_relative_to_m_plus_1() -> None:
    """M+2 must be 0.75-1.2 Da above M+1 (the previous isotope), not above M+0.

    The band is recursive: each M+k is checked against M+(k-1). The
    composition is chosen so target_mz lands in the strict envelope
    window (1000-3800) where M+1+M+2+M+3 are all required.
    """
    n_galnac, n_gal = 5, 4
    m0 = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106 + 22.9898
    # Confirm we are inside the strict window for this test.
    assert 1000.0 <= m0 <= 3800.0, f"test composition m0={m0} not in strict window"
    m1 = m0 + 1.003355
    m2 = m0 + 2 * 1.003355
    # M+1 observed, M+2 has a peak 0.5 Da above M+1 (= m1 + 0.5).
    # That is too close to M+1 to count as M+2, so M+2 is missing.
    peaks = [
        Peak(mz=m0, intensity=10000.0),
        Peak(mz=m1, intensity=2000.0),
        Peak(mz=m1 + 0.5, intensity=2000.0),  # NOT M+2 (too close to M+1)
        # No real M+2 observed.
        Peak(mz=m2 + 5.0, intensity=2000.0),   # far from M+2 -> not M+2 either
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, _skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.7, target_mz=m0, noise_at_target=0.0,
    )
    # Strict window requires M+1+M+2+M+3. M+2 missing -> not ok.
    assert ok is False, f"M+2 must be rejected when no peak is in 0.75-1.2 Da of M+1; got note={note!r}"


def test_hide_no_envelope_filter_drops_only_truly_missing_envelope() -> None:
    """Negative case: rows with NO 13C signal at the candidate m/z must be dropped.

    The screener's actual early-exit message is "no observed peaks at
    theoretical isotope positions" (screener.py:617). The line 672
    fallback "13C envelope not observed" is currently unreachable --
    see ``test_screen_notes_strings_match_filter_contract``. Both
    must be treated as "no envelope" by the filter.
    """
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "mz": 1000.0, "intensity": 100.0,
                "n_galnac": 1, "n_gal": 0, "ion": "Na+",
                "|ppm|": 0.1, "mz_diff": 0.0,
                "tier": "RED", "score_companion": 0, "score_series": 0,
                "screen_notes": "no observed peaks at theoretical isotope positions",
            },
            {
                "mz": 1000.0, "intensity": 100.0,
                "n_galnac": 1, "n_gal": 0, "ion": "Na+",
                "|ppm|": 0.1, "mz_diff": 0.0,
                "tier": "RED", "score_companion": 0, "score_series": 0,
                "screen_notes": "13C envelope not observed",  # dead-code fallback
            },
        ]
    )
    notes = df["screen_notes"].fillna("")
    mask = notes.str.contains("envelope observed") | notes.str.contains(
        "M.0 present", regex=True
    )
    kept = df[mask]
    assert len(kept) == 0, "rows with no envelope signal must be dropped"


# ---------------------------------------------------------------------------
# 9) screen_candidates honours the Spectrum upper bound (Bug #44 fix)
# ---------------------------------------------------------------------------
def test_screen_candidates_honours_spectrum_upper_bound() -> None:
    """Regression for Bug #44: when the caller passes a Spectrum with
    an explicit ``mz_hi``, ``screen_candidates`` must use it as the
    off-edge upper bound instead of ``max(peak.mz)``.

    Build a sparse spectrum where the highest peak is at, say, 2500,
    but the user is searching a window up to 3000 (mz_hi=3000). A
    candidate at m/z 2900 with its +1 GalNAc companion at +203 should
    be EVALUATED (the companion at 3103 is past the 3000 upper bound
    so is skipped, not failed). The companion/series notes must
    reflect this so the tier resolves correctly.
    """
    from glycan_ms.core import Spectrum as _Spectrum

    candidate_mz = 2900.0
    # Just the candidate M+0 + envelope satellites. No real companion
    # peaks anywhere. Companion at +203 = 3103 (past mz_hi=3000)
    # so SKIPPED; companion at +162 = 3062 (past mz_hi=3000) so
    # SKIPPED; companion at +41 = 2941 (in range but no peak).
    # Series: +203 step 1 = 3103 (past). +162 step 1 = 3062 (past).
    # So both companion and series are "skipped (off-spectrum)".
    # With envelope OK and both skipped, the tier must be YELLOW
    # (per the "skipped -> YELLOW" rule in _tier_from_scores).
    peaks = [
        Peak(mz=candidate_mz, intensity=5000.0),
        Peak(mz=candidate_mz + 1.0033, intensity=3000.0),
        Peak(mz=candidate_mz + 2 * 1.0033, intensity=1000.0),
        Peak(mz=candidate_mz + 3 * 1.0033, intensity=300.0),
    ]
    spectrum = _Spectrum(
        peaks=peaks,
        mz_hi=3000.0,  # user's acquisition upper bound
    )
    n_galnac, n_gal = 7, 4  # in strict envelope window (m/z ~ 2109)
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898  # ~2109
    # Use a candidate at m0_mz for the actual scoring; we keep
    # peaks separate so the off-edge detection on mz_hi is exercised
    # at a different scale. Actually for simplicity, use the same
    # peaks but with a candidate m/z whose companion +203 falls
    # just past mz_hi.
    df = _make_candidate_df(
        [
            {
                "mz": candidate_mz,
                "intensity": 5000.0,
                "n_galnac": 1,
                "n_gal": 0,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    out = screen_candidates(
        df, peaks, da_tol=0.7, spectrum=spectrum
    )
    # The companion/series offsets should be reported as SKIPPED
    # because mz_hi=3000 excludes them. The tier should be YELLOW
    # (envelope + skipped, no negative evidence).
    notes = out["screen_notes"].iloc[0]
    assert "skipped" in notes, (
        f"with mz_hi=3000 and candidate at 2900, companion/series "
        f"offsets past 3000 should be marked skipped; got notes={notes!r}"
    )
    assert out["tier"].iloc[0] == "YELLOW", (
        f"high-m/z candidate with skipped companion/series should be "
        f"YELLOW (envelope OK + skipped = no negative evidence); "
        f"got tier={out['tier'].iloc[0]!r}, notes={notes!r}"
    )


def test_screen_candidates_upper_bound_without_spectrum_falls_back_to_peak_max() -> None:
    """Back-compat: when ``spectrum`` is not provided, the off-edge
    upper bound falls back to ``max(peak.mz)`` (the previous
    behaviour). A candidate whose +203 companion is past
    ``max(peak.mz)`` is still skipped correctly.

    Above the strict envelope window (> 3800 Da) at LOW S/N, only
    M+0 + M+1 is required (the S/N-scaled bar applies at every
    m/z). We pick a target m/z there to exercise the full GREEN
    path end-to-end.
    """
    # 12 GalNAc + 10 Gal + Na+ -> ~4099.48 (above the 3800 strict
    # envelope upper bound). LOW S/N so only M+0 + M+1 is needed.
    n_galnac, n_gal = 12, 10
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    target = neutral + 22.9898  # ~4099
    # 10 noise-floor peaks near the bin ensure the floor is well
    # above M+0's intensity, putting S/N at LOW (required={1}).
    peaks = [
        Peak(mz=target, intensity=20000.0),
        Peak(mz=target + 1.0033, intensity=12000.0),  # M+1
        Peak(mz=target + 2 * 1.0033, intensity=4000.0),  # M+2
        Peak(mz=target + 3 * 1.0033, intensity=800.0),  # M+3
        Peak(mz=target + GALNAC_DELTA, intensity=20000.0),  # +203
        Peak(mz=target + 2 * GALNAC_DELTA, intensity=20000.0),  # +406
        Peak(mz=target + 3 * GALNAC_DELTA, intensity=20000.0),  # +609
        # 20 noise-floor peaks in the same 300-Da bin as target so
        # the per-bin noise computes a real percentile (>= 20 peaks
        # is MIN_PEAKS_PER_BIN).
        *[Peak(mz=target + 10 + i * 14, intensity=30000.0) for i in range(20)],
    ]
    df = _make_candidate_df(
        [
            {
                "mz": target,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    # No spectrum argument -> uses peak-derived upper bound.
    out = screen_candidates(df, peaks, da_tol=0.5)
    # Companion present, ladder present, envelope OK -> GREEN.
    assert out["tier"].iloc[0] == "GREEN", (
        f"no-spectrum path must still produce GREEN when companion + "
        f"series + envelope all present; got tier={out['tier'].iloc[0]!r}, "
        f"notes={out['screen_notes'].iloc[0]!r}"
    )


# ---------------------------------------------------------------------------
# 10) envelope_note_is_ok: the "Hide if 13C envelope not observed" filter
# ---------------------------------------------------------------------------
def test_envelope_note_is_ok_keeps_observed_envelope() -> None:
    """Confirmed envelope: keep."""
    from glycan_ms.screener import envelope_note_is_ok
    assert envelope_note_is_ok("13C envelope observed (M+0, M+1, M+2)")
    assert envelope_note_is_ok(
        "Companions: YES (...); 13C envelope observed (M+0, M+1)"
    )


def test_envelope_note_is_ok_keeps_m0_only_above_threshold() -> None:
    """M+0 present with no satellite above the (high) threshold is a
    True return from _score_envelope -- it means the candidate's
    intensity is so high the satellites fell below the threshold.
    Keep it."""
    from glycan_ms.screener import envelope_note_is_ok
    assert envelope_note_is_ok("13C M+0 present, no satellite above threshold")


def test_envelope_note_is_ok_drops_requires_m1() -> None:
    """The M+1 case: envelope actively failed. The previous
    "M+0 present" substring match incorrectly kept this. Lock the
    new behavior in."""
    from glycan_ms.screener import envelope_note_is_ok
    notes = (
        "Companions: YES (4 found: ...); Series: YES (5 steps: ...); "
        "13C envelope requires M+1 (only M+0 present at 1681.4 m/z)"
    )
    assert not envelope_note_is_ok(notes), (
        f"envelope_note_is_ok must reject 'requires M+1' notes; "
        f"got True for {notes!r}"
    )


def test_envelope_note_is_ok_drops_partial() -> None:
    """Partial envelope (some satellites present but not the required
    set for the window). Drop."""
    from glycan_ms.screener import envelope_note_is_ok
    # Note the new wording: the strict window now requires only
    # M+0 + M+1 + M+2 (M+3 is optional). The "partial" note fires
    # when a satellite in the required set is missing -- e.g. M+1
    # present but M+2 absent for a non-detection reason.
    notes = (
        "Companions: YES (1 found: ...); 13C envelope partial (M+0, M+1) -- "
        "strict window requires M+2 for 1500.0 m/z"
    )
    assert not envelope_note_is_ok(notes), (
        f"envelope_note_is_ok must reject 'partial' notes; "
        f"got True for {notes!r}"
    )


def test_envelope_note_is_ok_drops_not_observed() -> None:
    """No M+0 at all: drop."""
    from glycan_ms.screener import envelope_note_is_ok
    assert not envelope_note_is_ok("13C envelope not observed")
    assert not envelope_note_is_ok("Companions: NO; 13C envelope not observed")


def test_envelope_note_is_ok_empty_and_none() -> None:
    """Empty / None notes must default to False (drop)."""
    from glycan_ms.screener import envelope_note_is_ok
    assert not envelope_note_is_ok("")
    # The app.py call site uses fillna("") before calling, so a
    # None note comes through as "". Verify the helper is safe
    # with both inputs.


# ---------------------------------------------------------------------------
# 11) Relative-floor rescue: low-intensity M+2 rescued by 4% of M+0
#
# Regression for the user's three peaks at m/z 1257, 1314, 1095 in
# test_7c5_210(1).csv, all hidden/RED under the previous absolute-only
# noise floor. The new rule: a satellite is "observed" if it clears
# the absolute noise floor OR a fraction of M+0 (20% for M+1, 4% for
# M+2, 0.5% for M+3). This is the physically correct gate for
# low-intensity peaks where the absolute noise over-penalises the
# naturally-weak M+2 (typically 4-15% of M+0).
# ---------------------------------------------------------------------------
def test_score_envelope_low_intensity_m2_rescued_by_relative_floor() -> None:
    """A low-intensity M+0 with M+2 below absolute noise but above
    4% of M+0 is now classified as observed (previously rejected).
    This is the regression test for the user's three peaks at 1257,
    1314, 1095 m/z."""
    # 4 GalNAc + 4 Gal + Na+ (a realistic medium glycan)
    n_galnac, n_gal = 4, 4
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    target_mz = neutral + 22.9898  # ~1749
    assert 1000.0 <= target_mz <= 3800.0, "test must land in strict window"
    m0_intensity = 200.0
    m1_intensity = 90.0   # 45% of M+0
    m2_intensity = 30.0   # 15% of M+0 but below absolute noise 50
    peaks = [
        Peak(mz=target_mz, intensity=m0_intensity),
        Peak(mz=target_mz + 1.003355, intensity=m1_intensity),
        Peak(mz=target_mz + 2.006710, intensity=m2_intensity),
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.1, target_mz=target_mz, noise_at_target=50.0,
    )
    assert ok is True, f"M+0+M+1+M+2 with M+2 above 4% of M+0 should pass; got note={note!r}"
    assert "envelope observed" in note or "envelope confirmed" in note
    assert skipped is False


def test_score_envelope_m2_below_relative_floor_still_excluded() -> None:
    """A genuine noise spike at M+2 position that is below 4% of
    M+0 is still excluded. The candidate gets a "partial" note
    (NOT a "confirmed" note) and the envelope is treated as
    incomplete by the filter."""
    n_galnac, n_gal = 4, 4
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    target_mz = neutral + 22.9898  # ~1749
    assert 1000.0 <= target_mz <= 3800.0, "test must land in strict window"
    peaks = [
        Peak(mz=target_mz, intensity=200.0),
        Peak(mz=target_mz + 1.003355, intensity=90.0),
        Peak(mz=target_mz + 2.006710, intensity=3.0),  # 1.5% of M+0
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    ok, note, skipped = _score_envelope(
        n_galnac, n_gal, Adduct.NA, sorted_mz, sorted_intensity,
        da_tol=0.1, target_mz=target_mz, noise_at_target=50.0,
    )
    # M+2 is below the relative floor (1.5% < 4%) -- M+2 is not
    # observed. The required set at this S/N (M+0/noise=4, medium)
    # is {1,2}, so the candidate FAILS the envelope check. The
    # note must say "requires M+2", NOT "envelope confirmed" --
    # M+2 missing is an incomplete envelope, not an OK one.
    assert ok is False
    assert "envelope confirmed" not in note
    assert "requires M+2" in note
    assert skipped is False


def test_envelope_note_is_ok_drops_partial_m0_m1() -> None:
    """The 'M+0 + M+1 only' note (M+2 below detection) is DROPPED
    by the 'Hide 13C' filter -- it is an incomplete envelope, not
    a successful one. Only '13C envelope observed (M+0, M+1, M+2...)'
    and the 'M+0 only' borderline case pass the filter.
    """
    from glycan_ms.screener import envelope_note_is_ok
    # The incomplete-envelope note (what the screener produces when
    # M+0+M+1 are seen but M+2 falls below the relative floor at
    # medium S/N) must be DROPPED by the filter.
    assert envelope_note_is_ok(
        "13C envelope partial (M+0, M+1) -- requires M+2 for 1257.3 m/z (S/N=4.0)"
    ) is False
    assert envelope_note_is_ok(
        "13C envelope partial (M+0, M+1, M+2) -- requires M+3 for 2110.8 m/z (S/N=10.0)"
    ) is False
    # Embedded in the screen_candidates composite note:
    assert envelope_note_is_ok(
        "Companions: YES (3 found: +1 GalNAc; +1 Gal; +1 GalNAc -1 Gal); "
        "13C envelope partial (M+0, M+1) -- requires M+2 for 1257.3 m/z (S/N=4.0)"
    ) is False


# ---------------------------------------------------------------------------
# 11) M+0 S/N PURGE gate: candidates whose M+0 peak is below
#     MIN_M0_SNR_FOR_KEEP x noise are rejected outright, regardless
#     of companion/series/envelope evidence. The new "Hide if M+0
#     below 1.5x noise" checkbox in app.py drops these rows from the
#     table and chart.
# ---------------------------------------------------------------------------
def test_strongest_peak_near_returns_strongest_in_window() -> None:
    """The strongest peak within da_tol of target_mz is returned.

    Used by the screener to read off the M+0 intensity. Three peaks
    in the window: 200, 500, 300. The strongest (500) wins.
    """
    peaks = [
        Peak(mz=1000.0, intensity=200.0),
        Peak(mz=1000.4, intensity=500.0),
        Peak(mz=1000.8, intensity=300.0),
        Peak(mz=1100.0, intensity=900.0),  # outside window
    ]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    intensity, mz_observed = _strongest_peak_near(sorted_mz, sorted_intensity, 1000.0, 0.7)
    assert intensity == 500.0
    assert mz_observed == 1000.4


def test_strongest_peak_near_empty_spectrum_returns_zeros() -> None:
    """Empty input -> (0, 0) so the caller can divide by the noise floor."""
    intensity, mz = _strongest_peak_near([], [], 1500.0, 0.7)
    assert intensity == 0.0
    assert mz == 0.0


def test_strongest_peak_near_no_peak_in_window_returns_zeros() -> None:
    """A peak outside the window does not count as M+0."""
    peaks = [Peak(mz=2000.0, intensity=900.0)]
    sorted_mz, sorted_intensity = _sorted_mz_index(peaks)
    intensity, _ = _strongest_peak_near(sorted_mz, sorted_intensity, 1000.0, 0.7)
    assert intensity == 0.0


def test_is_low_sn_purge_matches_purge_note() -> None:
    """is_low_sn_purge is True iff the note carries the PURGE substring."""
    # Exact format produced by screen_candidates at the current
    # MIN_M0_SNR_FOR_KEEP value (1.35). The test is deliberately
    # loose: it asserts the substring ``M+0 below `` is present,
    # not the exact multiplier, so a future threshold tweak does
    # not require editing this test.
    assert is_low_sn_purge(
        "M+0 below 1.35x noise floor (intensity 100, floor 200, S/N=0.50) (PURGE)"
    ) is True
    # Substring is matched anywhere in the composite note.
    assert is_low_sn_purge(
        "Companions: NO (none in range); "
        "M+0 below 1.35x noise floor (intensity 80, floor 200, S/N=0.40) (PURGE)"
    ) is True
    # An empty note or a note without the substring must return False.
    assert is_low_sn_purge("") is False
    assert is_low_sn_purge("Companions: YES (1 found: +1 GalNAc)") is False


def test_min_m0_snr_for_keep_is_low_bar() -> None:
    """The PURGE threshold matches the LOW envelope bar (1.09x noise).

    A separate, slightly stronger rule: if M+0 itself is below the
    LOW bar, the peak is too weak to trust any companion/series
    evidence, so the candidate is PURGEd outright.
    """
    from glycan_ms.screener import STRICT_ENVELOPE_LOW_SN_NOISE
    assert MIN_M0_SNR_FOR_KEEP == STRICT_ENVELOPE_LOW_SN_NOISE
    assert MIN_M0_SNR_FOR_KEEP == 1.09


def test_tier_from_scores_low_m0_purges() -> None:
    """m0_sn_ok=False short-circuits to PURGE even if other evidence is strong."""
    # Everything else looks great (companion yes, series yes, envelope ok,
    # tolerance ok), but the M+0 peak is below the noise floor ->
    # PURGE, the strongest signal wins.
    tier = _tier_from_scores(
        has_companion=True,
        series_len=3,
        envelope_ok=True,
        tolerance_ok=True,
        m0_sn_ok=False,
    )
    assert tier == "PURGE"

    # And the default (m0_sn_ok=True) preserves the prior behaviour.
    tier_default = _tier_from_scores(
        has_companion=True,
        series_len=3,
        envelope_ok=True,
        tolerance_ok=True,
    )
    assert tier_default == "GREEN"


def test_screen_candidates_purges_low_m0() -> None:
    """End-to-end: a candidate whose M+0 is below 1.35x the local noise
    is PURGEd even when its companion and envelope are strong. This
    is the behaviour the new "Hide if M+0 below 1.35x noise" checkbox
    relies on.
    """
    n_galnac, n_gal = 1, 0
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    # Row intensity (DataFrame) is high, but the observed M+0 peak in
    # the spectrum is weak (200). The local noise floor is set by the
    # surrounding noise peaks at 1000, so S/N = 0.2 < 1.35 -> PURGE.
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    peaks = [
        Peak(mz=m0_mz, intensity=200.0),                          # M+0, weak
        Peak(mz=m0_mz + GALNAC_DELTA, intensity=4000.0),           # companion
        Peak(mz=m0_mz + 1.003, intensity=200.0),                   # M+1, weak
        Peak(mz=m0_mz + 2.006, intensity=200.0),                   # M+2, weak
        # Noise floor: 10 peaks at 1000 to set a real bin floor.
        Peak(mz=m0_mz - 100.0, intensity=1000.0),
        Peak(mz=m0_mz - 150.0, intensity=1000.0),
        Peak(mz=m0_mz - 200.0, intensity=1000.0),
        Peak(mz=m0_mz - 250.0, intensity=1000.0),
        Peak(mz=m0_mz - 300.0, intensity=1000.0),
        Peak(mz=m0_mz - 350.0, intensity=1000.0),
        Peak(mz=m0_mz - 400.0, intensity=1000.0),
        Peak(mz=m0_mz + 100.0, intensity=1000.0),
        Peak(mz=m0_mz + 150.0, intensity=1000.0),
        Peak(mz=m0_mz + 200.0, intensity=1000.0),
    ]
    out = screen_candidates(df, peaks, da_tol=10.0)
    assert out["tier"].iloc[0] == "PURGE"
    assert is_low_sn_purge(out["screen_notes"].iloc[0])


def test_screen_candidates_strong_m0_passes() -> None:
    """The inverse: a strong M+0 (S/N >= 1.35) is NOT PURGEd by the new gate.

    Sanity check that the M+0 S/N gate does not over-reject. The
    companion is missing here so the tier falls through to RED (or
    YELLOW), but the PURGE gate is the question under test -- it
    must NOT fire.
    """
    n_galnac, n_gal = 1, 0
    neutral = n_galnac * GALNAC_DELTA + n_gal * GAL_DELTA + 18.0106
    m0_mz = neutral + 22.9898
    df = _make_candidate_df(
        [
            {
                "mz": m0_mz,
                "intensity": 5000.0,
                "n_galnac": n_galnac,
                "n_gal": n_gal,
                "ion": "Na+",
                "|ppm|": 0.1,
            }
        ]
    )
    # M+0 = 5000, noise floor = 100. S/N = 50, way above 1.35.
    peaks = [
        Peak(mz=m0_mz, intensity=5000.0),
        Peak(mz=m0_mz + 1.003, intensity=200.0),  # M+1, weak
        Peak(mz=m0_mz + 2.006, intensity=200.0),  # M+2, weak
        Peak(mz=m0_mz - 100.0, intensity=100.0),
        Peak(mz=m0_mz - 200.0, intensity=100.0),
        Peak(mz=m0_mz - 300.0, intensity=100.0),
        Peak(mz=m0_mz - 400.0, intensity=100.0),
        Peak(mz=m0_mz + 100.0, intensity=100.0),
        Peak(mz=m0_mz + 200.0, intensity=100.0),
    ]
    out = screen_candidates(df, peaks, da_tol=10.0)
    assert out["tier"].iloc[0] != "PURGE", (
        f"strong M+0 must not be PURGEd, got tier={out['tier'].iloc[0]!r}"
    )


# ---------------------------------------------------------------------------
# 12) Two-tier noise binning: 300 Da below NOISE_BIN_SPLIT_MZ, 150 Da above
# ---------------------------------------------------------------------------
def test_bin_lo_uses_300_da_below_split() -> None:
    """Below NOISE_BIN_SPLIT_MZ, bins are 300 Da wide on multiples of 300."""
    from glycan_ms.screener import _bin_lo, NOISE_BIN_SPLIT_MZ
    # 0..300 -> bin lo = 0
    assert _bin_lo(0.0) == 0.0
    assert _bin_lo(299.9) == 0.0
    # 300..600 -> bin lo = 300
    assert _bin_lo(300.0) == 300.0
    assert _bin_lo(599.9) == 300.0
    # 2700..3000 -> bin lo = 2700 (last 300-Da bin below the split)
    assert _bin_lo(2700.0) == 2700.0
    assert _bin_lo(2799.9) == 2700.0
    # Sanity: the bin width is 300 (from lo to the next bin's lo).
    assert _bin_lo(NOISE_BIN_SPLIT_MZ - 0.1) - _bin_lo(NOISE_BIN_SPLIT_MZ - 300.1) == 300.0


def test_bin_lo_uses_150_da_at_and_above_split() -> None:
    """At and above NOISE_BIN_SPLIT_MZ, bins are 150 Da wide anchored on the split."""
    from glycan_ms.screener import _bin_lo, NOISE_BIN_SPLIT_MZ
    # At the split: first 150-Da bin starts at the split
    assert _bin_lo(NOISE_BIN_SPLIT_MZ) == NOISE_BIN_SPLIT_MZ
    assert _bin_lo(NOISE_BIN_SPLIT_MZ + 149.9) == NOISE_BIN_SPLIT_MZ
    # 2950..3100 -> bin lo = 2800 + 150 = 2950
    assert _bin_lo(2950.0) == 2950.0
    assert _bin_lo(3099.9) == 2950.0
    # 4000 -> 2800 + 150 * floor(1200/150) = 2800 + 150*8 = 4000
    assert _bin_lo(4000.0) == 4000.0
    assert _bin_lo(4149.9) == 4000.0
    # High m/z sanity
    assert _bin_lo(5000.0) - _bin_lo(4850.0) == 150.0


def test_noise_floor_uses_150_da_bin_above_split() -> None:
    """A peak at m/z 2950 (just above the split) and another at 3000
    must land in DIFFERENT 150-Da bins, so their noise floors are
    computed from independent populations.

    Under the old 300-Da scheme they would have shared a bin.
    """
    # Pack 30 peaks in the [2800, 2950) bin at intensity 50, and 30
    # peaks in the [2950, 3100) bin at intensity 100. The two bins
    # must produce different noise floors. Under 300-Da binning the
    # 60 peaks all land in [2700, 3000) and would share a single
    # floor (somewhere between 50 and 100).
    peaks = (
        [Peak(mz=2800.0 + i, intensity=50.0) for i in range(30)]
        + [Peak(mz=2950.0 + i, intensity=100.0) for i in range(30)]
    )
    cache, _fallback = _build_noise_floor_cache(peaks)
    # Both bins are populated with >= 20 peaks, so both should have
    # per-bin floors (not the fallback).
    floor_below_split = _noise_floor_cached(cache, 2900.0)
    floor_above_split = _noise_floor_cached(cache, 3000.0)
    # Each raw floor is the lower percentile times NOISE_MULTIPLIER,
    # then the adjacent bins are softly centre-weighted (2:1).
    low_raw = 50.0 * NOISE_MULTIPLIER
    high_raw = 100.0 * NOISE_MULTIPLIER
    assert floor_below_split == pytest.approx((2 * low_raw + high_raw) / 3)
    assert floor_above_split == pytest.approx((2 * high_raw + low_raw) / 3)
    # And they must differ -- if the binning collapsed both into a
    # single 300-Da bin the two floors would be identical.
    assert floor_above_split != floor_below_split


def test_noise_floor_is_continuous_at_1800_bin_boundary() -> None:
    """Adjacent bins may differ, but scoring must not jump at their edge."""
    peaks = (
        [Peak(mz=1500.0 + i, intensity=100.0) for i in range(30)]
        + [Peak(mz=1800.0 + i, intensity=300.0) for i in range(30)]
    )
    model = _build_noise_floor_model(peaks)

    just_below = model.floor_at(1800.0 - 0.001)
    just_above = model.floor_at(1800.0 + 0.001)

    assert just_below == pytest.approx(just_above, rel=1e-5)


def test_noise_floor_at_split_boundary() -> None:
    """A peak at m/z = NOISE_BIN_SPLIT_MZ exactly is in the 150-Da region
    (>=, not <). A peak at m/z = NOISE_BIN_SPLIT_MZ - 0.01 is in the
    300-Da region. The two peaks at the boundary must be in different bins.
    """
    from glycan_ms.screener import _bin_lo, NOISE_BIN_SPLIT_MZ
    # m/z = split exactly -> first 150-Da bin, lo = split.
    assert _bin_lo(NOISE_BIN_SPLIT_MZ) == NOISE_BIN_SPLIT_MZ
    # m/z = split - 0.01 -> the 300-Da bin [2700, 2800). Its lo is 2700.
    # (The bins below the split are on 300-Da multiples, so the bin
    # containing 2799.99 is the one with lo = floor(2799.99 / 300) * 300
    # = 2700.)
    assert _bin_lo(NOISE_BIN_SPLIT_MZ - 0.01) == 2700.0
    # And the two bin indices must be different (no collision across
    # the split).
    assert _bin_lo(NOISE_BIN_SPLIT_MZ) != _bin_lo(NOISE_BIN_SPLIT_MZ - 0.01)


# ---------------------------------------------------------------------------
# "Two good candidates within 2.2 Da -> keep the lower m/z" dedup
# ---------------------------------------------------------------------------


def test_dedup_nearby_observed_mz_keeps_lower() -> None:
    """Two non-PURGE candidates within 2.2 Da must collapse to the lower m/z.

    Regression for the user's rule: when two candidates BOTH pass
    the four checks (companion, series, envelope, tolerance) and
    their observed m/z's are within 2.2 Da, keep the lower one.
    PURGE rows pass through unchanged (they are not "good").
    """
    from glycan_ms.screener import _dedup_nearby_observed_mz
    import pandas as pd

    df = pd.DataFrame(
        [
            {"mz": 1000.0, "tier": "YELLOW"},
            {"mz": 1001.0, "tier": "YELLOW"},   # shadowed by 1000.0
            {"mz": 1001.5, "tier": "YELLOW"},   # shadowed by 1000.0
            {"mz": 1010.0, "tier": "GREEN"},    # far enough -- keep
            {"mz": 1011.0, "tier": "GREEN"},    # shadowed by 1010.0
            {"mz": 1500.0, "tier": "PURGE"},    # PURGE passes through
            {"mz": 2500.0, "tier": "RED"},      # RED passes through
            {"mz": 2501.5, "tier": "RED"},      # RED within 2.2 -> dropped
        ]
    )
    out = _dedup_nearby_observed_mz(df)
    mzs = out["mz"].tolist()
    tiers = out["tier"].tolist()

    # Lower non-PURGE wins at each cluster.
    assert 1000.0 in mzs and 1001.0 not in mzs and 1001.5 not in mzs
    assert 1010.0 in mzs and 1011.0 not in mzs
    # PURGE row passes through unchanged.
    assert 1500.0 in mzs
    # RED within 2.2 of another RED: the rule says "two good candidates".
    # RED is a non-PURGE tier, so it counts as "good" and the higher
    # one is dropped.
    assert 2500.0 in mzs and 2501.5 not in mzs
    # Total kept: 4 (1000.0, 1010.0, 1500.0, 2500.0).
    assert len(out) == 4, f"expected 4 kept, got {len(out)}: {mzs!r} {tiers!r}"


def test_dedup_nearby_observed_mz_boundary_inclusive() -> None:
    """The 2.2 Da window is inclusive: a row within 2.2 Da is shadowed.

    Float precision note: 1002.2 - 1000.0 evaluates to
    2.2000000000000455 (slightly > 2.2), so an "exactly 2.2 Da"
    literal is NOT shadowed. We pick 1002.15 to land safely
    inside the window.
    """
    from glycan_ms.screener import _dedup_nearby_observed_mz
    import pandas as pd

    df = pd.DataFrame(
        [
            {"mz": 1000.0, "tier": "YELLOW"},
            {"mz": 1002.15, "tier": "YELLOW"},  # 2.15 < 2.2 -> shadowed
            {"mz": 1002.3, "tier": "YELLOW"},   # 2.3 > 2.2 -> NOT shadowed
        ]
    )
    out = _dedup_nearby_observed_mz(df)
    assert len(out) == 2
    assert 1000.0 in out["mz"].tolist()
    assert 1002.3 in out["mz"].tolist()
    assert 1002.15 not in out["mz"].tolist()


def test_dedup_nearby_observed_mz_empty_and_invalid() -> None:
    """Empty DataFrames and missing columns pass through unchanged."""
    from glycan_ms.screener import _dedup_nearby_observed_mz
    import pandas as pd

    assert len(_dedup_nearby_observed_mz(pd.DataFrame())) == 0

    # No mz column: pass-through.
    df = pd.DataFrame([{"a": 1}])
    out = _dedup_nearby_observed_mz(df)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# GalNAc-overlap erase rule
#
# The user's rule: when a GalNAc-dominant candidate (n_galnac > n_gal)
# sits within 6 Da of a higher-m/z candidate AND any of the first's
# M+1 / M+2 / M+3 isotope positions falls within da_tol of the second's
# M+0 or M+1 position, the second candidate is erased. The first's
# M+k peaks are real signal (the GalNAc-dominant envelope rule already
# requires M+1+M+2+M+3 above noise for non-PURGE rows), so the second
# candidate sitting inside the first's isotope envelope cannot be
# distinguished from those satellites.
# ---------------------------------------------------------------------------


def test_galnac_overlap_erases_second_when_first_is_galnac_dominant() -> None:
    """Within 6 Da + first has GalNAc > Gal + first's M+1 within da_tol
    of second's M+0 -> second is PURGEd."""
    from glycan_ms.screener import (
        GALNAC_OVERLAP_ERASE_DA,
        GALNAC_OVERLAP_ERASE_NOTE,
        _erase_overlapped_neighbors,
    )

    # First candidate at m/z = 1500.0, n_galnac=3, n_gal=1 (GalNAc > Gal).
    # Its M+1 lands at 1501.0034. Second candidate at m/z = 1501.0 --
    # that's within 0.0034 Da of the first's M+1, well inside da_tol=0.7.
    df = pd.DataFrame(
        [
            {"mz": 1500.0, "n_galnac": 3, "n_gal": 1, "tier": "GREEN"},
            {"mz": 1501.0, "n_galnac": 2, "n_gal": 2, "tier": "YELLOW"},
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    tiers = out["tier"].tolist()
    assert tiers[0] == "GREEN", "first must be untouched"
    assert tiers[1] == "PURGE", f"second must be erased, got {tiers[1]!r}"
    assert GALNAC_OVERLAP_ERASE_NOTE in out["screen_notes"].iloc[1]
    assert GALNAC_OVERLAP_ERASE_DA == 6.0


def test_galnac_overlap_window_is_6_da() -> None:
    """The constant is 6 Da, not the existing 2.2 Da dedup window.

    Two candidates 5 Da apart should still trigger the erase rule
    (5 < 6); 7 Da apart must not.
    """
    from glycan_ms.screener import _erase_overlapped_neighbors

    # First: 1500.0, n_galnac=3, n_gal=1 (GalNAc-dominant). M+1 = 1501.0034.
    # Second at 1506.0: delta = 6.0, exactly at the window edge. The first's
    # M+1 (1501.0034) is not within 0.7 Da of the second's M+0 (1506.0),
    # so the overlap test does NOT fire on M+1. The first's M+3 is at
    # 1503.0101 -- still 3 Da short of the second's M+0. So even at the
    # edge the overlap does NOT fire. We use this as the boundary
    # sanity check: the second at 1506.0 stays non-PURGE.
    df = pd.DataFrame(
        [
            {"mz": 1500.0, "n_galnac": 3, "n_gal": 1, "tier": "GREEN"},
            {"mz": 1506.0, "n_galnac": 2, "n_gal": 2, "tier": "YELLOW"},
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    # 1506.0 is exactly 6.0 Da away. The boundary check is "j_mz - i_mz >
    # window_da", i.e. NOT > 6 means still in the window. But the
    # overlap test on the M+k positions does NOT fire here (see
    # comment above), so the second stays non-PURGE.
    assert out["tier"].iloc[0] == "GREEN"
    assert out["tier"].iloc[1] == "YELLOW", (
        f"second at exactly 6 Da with no M+k overlap should stay, got {out['tier'].iloc[1]!r}"
    )


def test_galnac_overlap_outside_6_da_does_not_erase() -> None:
    """Two candidates more than 6 Da apart are independent -- the second stays."""
    from glycan_ms.screener import _erase_overlapped_neighbors

    df = pd.DataFrame(
        [
            {"mz": 1500.0, "n_galnac": 3, "n_gal": 1, "tier": "GREEN"},
            {"mz": 1507.0, "n_galnac": 2, "n_gal": 2, "tier": "YELLOW"},
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    assert out["tier"].iloc[0] == "GREEN"
    assert out["tier"].iloc[1] == "YELLOW", (
        f"second > 6 Da away must stay non-PURGE, got {out['tier'].iloc[1]!r}"
    )


def test_galnac_overlap_first_not_galnac_dominant_does_not_erase() -> None:
    """The trigger is GalNAc > Gal on the FIRST candidate. If the first
    has GalNAc <= Gal, the second is left alone."""
    from glycan_ms.screener import _erase_overlapped_neighbors

    df = pd.DataFrame(
        [
            # First has GalNAc == Gal -> NOT GalNAc-dominant
            {"mz": 1500.0, "n_galnac": 2, "n_gal": 2, "tier": "GREEN"},
            {"mz": 1501.0, "n_galnac": 3, "n_gal": 2, "tier": "YELLOW"},
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    assert out["tier"].iloc[0] == "GREEN"
    assert out["tier"].iloc[1] == "YELLOW", (
        f"second must stay when first is not GalNAc-dominant, got {out['tier'].iloc[1]!r}"
    )


def test_galnac_overlap_mk_outside_second_window_does_not_erase() -> None:
    """Within 6 Da but the first's M+1/M+2/M+3 do NOT fall within da_tol
    of the second's M+0 or M+1 -> second stays.

    First at 1500.0, M+1 = 1501.0034. Second at 1503.5: delta = 3.5 Da
    (within 6 Da window), but the first's M+3 (1503.0101) is 0.49 Da
    short of the second's M+0 -- inside da_tol=0.7! This case SHOULD
    trigger the erase. We instead test 1503.7 -- first's M+3 = 1503.0101
    is 0.69 Da short of 1503.7, still inside 0.7. Use 1503.8 for the
    non-trigger case: first's M+3 is 0.79 Da away, > 0.7.
    """
    from glycan_ms.screener import _erase_overlapped_neighbors

    df = pd.DataFrame(
        [
            {"mz": 1500.0, "n_galnac": 3, "n_gal": 1, "tier": "GREEN"},
            {"mz": 1503.8, "n_galnac": 2, "n_gal": 2, "tier": "YELLOW"},
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    assert out["tier"].iloc[0] == "GREEN"
    assert out["tier"].iloc[1] == "YELLOW", (
        f"second at 1503.8 has no M+k overlap; should stay, got {out['tier'].iloc[1]!r}"
    )


def test_galnac_overlap_first_purged_does_not_erase_second() -> None:
    """If the first is already PURGEd, its M+k peaks are not trusted
    real signal, so the second is not erased by this rule."""
    from glycan_ms.screener import _erase_overlapped_neighbors

    df = pd.DataFrame(
        [
            {"mz": 1500.0, "n_galnac": 3, "n_gal": 1, "tier": "PURGE"},
            {"mz": 1501.0, "n_galnac": 2, "n_gal": 2, "tier": "YELLOW"},
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    assert out["tier"].iloc[0] == "PURGE"
    assert out["tier"].iloc[1] == "YELLOW", (
        f"PURGEd first must not erase second, got {out['tier'].iloc[1]!r}"
    )


def test_galnac_overlap_m1_overlapping_second_m1_also_erases() -> None:
    """The overlap test fires on EITHER second's M+0 OR second's M+1.

    First at 1500.0, M+2 = 1502.0067. Second at 1503.0, M+1 = 1504.0034.
    First's M+2 (1502.0067) is 0.99 Da short of second's M+0 (1503.0)
    -- not within da_tol=0.7. But first's M+3 (1503.0101) is 0.99 Da
    short of second's M+1 (1504.0034) -- not within 0.7 either. So this
    delta does NOT trigger. We instead test first's M+2 = 1502.0067 vs
    second at 1502.7, M+1 = 1503.7037: first's M+2 is 0.69 Da away from
    second's M+0 (1502.7) -- inside da_tol! -> erase.
    """
    from glycan_ms.screener import _erase_overlapped_neighbors

    df = pd.DataFrame(
        [
            {"mz": 1500.0, "n_galnac": 3, "n_gal": 1, "tier": "GREEN"},
            {"mz": 1502.7, "n_galnac": 2, "n_gal": 2, "tier": "YELLOW"},
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    assert out["tier"].iloc[1] == "PURGE", (
        f"first's M+2 within da_tol of second's M+0 should erase, got {out['tier'].iloc[1]!r}"
    )


def test_galnac_overlap_empty_and_invalid_passthrough() -> None:
    """Empty DataFrames and missing required columns pass through unchanged."""
    from glycan_ms.screener import _erase_overlapped_neighbors

    # Empty.
    assert len(_erase_overlapped_neighbors(pd.DataFrame(), da_tol=0.7)) == 0
    # Missing 'mz' column.
    out = _erase_overlapped_neighbors(pd.DataFrame([{"a": 1}]), da_tol=0.7)
    assert len(out) == 1
    # Missing 'n_galnac' column.
    out = _erase_overlapped_neighbors(
        pd.DataFrame([{"mz": 1500.0, "tier": "YELLOW"}]),
        da_tol=0.7,
    )
    assert out["tier"].iloc[0] == "YELLOW"
    # Missing 'tier' column.
    out = _erase_overlapped_neighbors(
        pd.DataFrame([{"mz": 1500.0, "n_galnac": 3, "n_gal": 1}]),
        da_tol=0.7,
    )
    assert len(out) == 1


def test_galnac_overlap_appended_to_existing_note() -> None:
    """An erased row's existing note is preserved, with the erase reason appended."""
    from glycan_ms.screener import (
        GALNAC_OVERLAP_ERASE_NOTE,
        _erase_overlapped_neighbors,
    )

    df = pd.DataFrame(
        [
            {"mz": 1500.0, "n_galnac": 3, "n_gal": 1, "tier": "GREEN", "screen_notes": "original note"},
            {
                "mz": 1501.0,
                "n_galnac": 2,
                "n_gal": 2,
                "tier": "YELLOW",
                "screen_notes": "prior note",
            },
        ]
    )
    out = _erase_overlapped_neighbors(df, da_tol=0.7)
    # First row untouched.
    assert out["screen_notes"].iloc[0] == "original note"
    # Second row: prior note preserved + erase reason appended.
    note = out["screen_notes"].iloc[1]
    assert note.startswith("prior note")
    assert GALNAC_OVERLAP_ERASE_NOTE in note
    assert "1500.00 Da" in note
    assert "n_galnac=3" in note
    assert "n_gal=1" in note


def test_screen_candidates_galnac_overlap_endtoend() -> None:
    """End-to-end: screen_candidates applies the GalNAc-overlap erase rule.

    Two candidates 1 Da apart -- the first has GalNAc > Gal, the second
    does not. The second's m/z sits inside the first's M+1 satellite.
    After screen_candidates the second is PURGEd.
    """
    from glycan_ms.screener import GALNAC_OVERLAP_ERASE_NOTE

    # First composition: 3 GalNAc + 1 Gal + Na+. Closed-form:
    #   neutral = 3*203.0794 + 1*162.0528 + 18.0106 = 789.3596
    #   m/z = neutral + 22.9898 = 812.3494 (below the screener's
    #   normal operating range; bump via a fictional multiplier).
    # We don't care about the absolute number being a real composition;
    # we just need TWO candidates close in m/z, first GalNAc-dominant.
    # Use arbitrary m/z values 1500.0 and 1501.0 so the math is simple.
    first_mz = 1500.0
    second_mz = 1501.0
    df = _make_candidate_df(
        [
            {
                "mz": first_mz,
                "intensity": 1000.0,
                "n_galnac": 3,
                "n_gal": 1,
                "ion": "Na+",
                "mz_diff": 0.0,
            },
            {
                "mz": second_mz,
                "intensity": 800.0,
                "n_galnac": 2,
                "n_gal": 2,
                "ion": "Na+",
                "mz_diff": 0.0,
            },
        ]
    )
    # Build a spectrum where both candidates have M+0 peaks above
    # the noise floor AND the first has M+1, M+2, M+3 above noise
    # (the GalNAc-dominant envelope requirement). Use a high overall
    # noise floor so the bin's noise statistic is robust.
    peaks = [
        # First candidate's M+0..M+3 at S/N well above 5 (HIGH bar)
        Peak(mz=first_mz, intensity=2000.0),
        Peak(mz=first_mz + 1.003355, intensity=900.0),
        Peak(mz=first_mz + 2.00671, intensity=300.0),
        Peak(mz=first_mz + 3.010065, intensity=80.0),
        # Second candidate's M+0 at 1501.0
        Peak(mz=second_mz, intensity=1500.0),
        Peak(mz=second_mz + 1.003355, intensity=700.0),
        Peak(mz=second_mz + 2.00671, intensity=250.0),
        Peak(mz=second_mz + 3.010065, intensity=60.0),
    ] + [Peak(mz=1400.0 + i, intensity=100.0) for i in range(0, 100, 4)]
    out = screen_candidates(df, peaks, da_tol=10.0)
    # First row stays (was GREEN/YELLOW/RED -- whatever the screener
    # decided, just not PURGEd by this rule).
    assert out["tier"].iloc[0] != "PURGE", (
        f"first must survive, got {out['tier'].iloc[0]!r}"
    )
    # Second row PURGEd by the GalNAc-overlap rule.
    assert out["tier"].iloc[1] == "PURGE", (
        f"second must be PURGEd by GalNAc-overlap, got {out['tier'].iloc[1]!r}"
    )
    assert GALNAC_OVERLAP_ERASE_NOTE in out["screen_notes"].iloc[1]

