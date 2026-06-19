"""Candidate screener for glycan MS composition hits.

Beyond the monoisotopic composition match produced by :mod:`glycan_ms.core`,
this module validates each candidate against three extra signals that are
characteristic of real O-glycan MS data:

1. **Companion peaks at fixed offsets.** If a candidate (n, m, adduct) at
   m/z X is real, we should often see peaks at X + 203.0794 (one extra
   GalNAc), X + 162.0528 (one extra Gal), X + 41.0266 (one extra GalNAc,
   one fewer Gal), X + 15.9739 (same composition, K+ instead of Na+), or
   combinations of these. Each offset class is checked independently and
   the count contributes to a score.

2. **Series / 3-in-a-row at a fixed offset.** A "ladder" of three or more
   additional peaks each separated by the same offset d, with each gap
   within 0.75 - 1.2 of d, all peaks above the windowed noise floor. This
   is the strongest single piece of evidence that a candidate is a real
   glycan series rather than a random mass coincidence.

3. **13C isotope envelope.** For each candidate, build the theoretical
   isotopologue distribution (M+0, M+1, M+2, ...) as a closed-form
   multinomial on the carbon isotope distribution and check that M+0
   and at least one of M+1/M+2 are observed above the noise floor in
   the spectrum.

All three checks are gated on a **windowed noise floor** -- an estimate
of the per-m/z-region background intensity. The user explicitly noted
that 2500-3000 m/z has a different noise level than 1400-1700, so the
floor is computed from a robust percentile of peak intensities in a
bin (default 300 Da wide) around the candidate, not a single global
number.

The public entry point is :func:`screen_candidates`, which matches the
call signature in :mod:`glycan_ms.app` (``app.py`` lines 175-215) and
attaches four new columns to the candidate DataFrame:

- ``tier``           -- ``GREEN`` / ``YELLOW`` / ``RED`` / ``PURGE``
- ``screen_notes``   -- human-readable reason(s) for the tier
- ``score_companion``-- count of companion offset classes found (0..N)
- ``score_series``   -- length of the longest detected chain (0..3+)

The tier mapping is:

================  =============  ===========  =========  ==========
Companion (>=1)   Series (>=3)   Envelope     Tolerance  Tier
================  =============  ===========  =========  ==========
yes               yes            ok           ok         GREEN
yes               no             ok           ok         YELLOW
no                yes            ok           ok         YELLOW
no                no             ok           ok         RED
any               any            any          bad        PURGE
================  =============  ===========  =========  ==========

If the envelope helper ever returns ``None`` (defensive guard for
unparseable compositions -- which should not occur for the
``C{x}H{y}N{n}O{w}`` closed form the screener generates), the envelope
column is silently skipped and the tier is decided on companion +
series + tolerance alone. This module never raises an exception on bad
input data; empty / malformed inputs yield empty / NaN outputs.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from .core import Adduct, Peak, Spectrum


# --- Mass offsets ----------------------------------------------------------

# Residue deltas (ExPASy / GlycoMod monoisotopic).
GALNAC_DELTA: Final[float] = 203.0794   # +1 GalNAc
GAL_DELTA: Final[float] = 162.0528      # +1 Gal
GALNAC_MINUS_GAL: Final[float] = 41.0266  # +1 GalNAc, -1 Gal

# Adduct pair deltas. Each adduct is monoisotopic, so the difference is
# the difference of the adduct masses.
K_MINUS_NA: Final[float] = 15.9739      # K+ in place of Na+
NA_MINUS_H: Final[float] = 21.9825      # Na+ in place of H+
K_MINUS_H: Final[float] = 37.9564       # K+ in place of H+

# Atom masses for the closed-form composition -> formula derivation.
# GalNAc residue: C8H13NO5 (203.0794 Da).
# Gal residue:    C6H10O5  (162.0528 Da).
# Free reducing-end water: H2O.
# So a composition of n GalNAc + m Gal + H2O yields:
#   C = 8n + 6m
#   H = 13n + 10m + 2
#   N = n
#   O = 5n + 5m + 1


# --- Companion-offset table ------------------------------------------------

# Single offsets (independent checks). A "hit" on any of these counts as
# one companion class. Combinations (e.g. +203+162) are checked below as
# a separate pass.
_SINGLE_OFFSETS: Final[tuple[tuple[float, str], ...]] = (
    (GALNAC_DELTA, "+1 GalNAc"),
    (GAL_DELTA, "+1 Gal"),
    (GALNAC_MINUS_GAL, "+1 GalNAc -1 Gal"),
    (K_MINUS_NA, "K+/Na+ ion pair"),
)

# Combinations: pairs of single offsets, since asking for +203+162 (one
# extra of each) or +203+15.97 (extra GalNAc with K+ instead of Na+) is
# the most informative test in practice. Higher-order combinations (e.g.
# +203+162+15.97) are not enumerated -- they require a candidate to
# already be well-supported by 1-offset and pair evidence, and the user
# did not ask for them.
_PAIR_OFFSETS: Final[tuple[tuple[float, str], ...]] = (
    (GALNAC_DELTA + GAL_DELTA, "+1 GalNAc +1 Gal"),
    (GALNAC_DELTA + K_MINUS_NA, "+1 GalNAc, K+/Na+"),
    (GAL_DELTA + K_MINUS_NA, "+1 Gal, K+/Na+"),
    (GALNAC_MINUS_GAL + K_MINUS_NA, "+1 GalNAc -1 Gal, K+/Na+"),
    (GALNAC_DELTA + NA_MINUS_H, "+1 GalNAc, Na+/H+"),
)

# Series-eligible offsets: those for which a "ladder" of 3+ peaks at the
# same spacing is meaningful. The ion-pair deltas are excluded because
# the ladder interpretation only makes sense for residue additions.
SERIES_OFFSETS: Final[tuple[tuple[float, str], ...]] = (
    (GALNAC_DELTA, "GalNAc ladder"),
    (GAL_DELTA, "Gal ladder"),
    (GALNAC_MINUS_GAL, "GalNAc-Gal alternation"),
)

# Gap tolerance: each step in a series must be within 0.75 to 1.2 of d.
SERIES_GAP_LO: Final[float] = 0.75
SERIES_GAP_HI: Final[float] = 1.20

# Isotope gap tolerance: each M+k must be 0.75 to 1.2 Da above the
# PREVIOUS isotope's theoretical m/z. The 13C shift is 1.003355 Da, so
# this band straddles the theoretical value with the same 0.75/1.2 of d
# shape as the series-ladder rule. A peak closer than 0.75 Da is not a
# 13C isotope; a peak further than 1.2 Da is more than one 13C
# substitution. Either way it does not count as an observed M+k.
ISOTOPE_GAP_LO: Final[float] = 0.75
ISOTOPE_GAP_HI: Final[float] = 1.20

# How many "additional" peaks after the candidate are required for a
# series to count. The user said "3 signals after that signal", so 3.
SERIES_MIN_LENGTH: Final[int] = 3

# Noise-floor bin width in Da. The spectrum is divided into two
# regions for noise estimation: 300 Da bins from 0 to NOISE_BIN_SPLIT_MZ,
# then 150 Da bins above. The wider bins at low m/z keep the
# per-bin noise statistic robust (>= MIN_PEAKS_PER_BIN=20 peaks per
# bin for the percentile to be meaningful); the narrower bins at
# high m/z give finer-grained local estimates where glycan peaks
# are sparser and adjacent compositions (e.g. +1 GalNAc) are 203 Da
# apart, so a 300-Da window would smooth over distinct candidates.
DEFAULT_NOISE_BIN_WIDTH: Final[float] = 300.0
#: m/z below which the 300-Da binning is used.
NOISE_BIN_SPLIT_MZ: Final[float] = 2800.0
#: m/z at and above which the 150-Da binning is used.
NOISE_BIN_WIDTH_HIGH: Final[float] = 150.0

# Companion/series search tolerance in Daltons. Wider than the
# candidate-match tolerance because a +162 (or +41 GalNAc-for-Gal
# substitution) peak is a real biosynthetic relationship, and a few-
# tenths of a Da of mass error is normal. 0.75 Da is the user's
# standing default: "increase the like by off tolerance to be like
# 0.75". Both _score_companion and _score_series use this constant
# (taken as max with the user's Da slider so the search is never
# narrower than either signal demands).
DEFAULT_COMPANION_TOL_DA: Final[float] = 0.75

# Companion "tight-offset" override: a peak inside the search
# window whose m/z is within this many Daltons of the
# theoretical companion position is treated as a real
# companion regardless of the local noise floor. The rationale
# is that a real biological signal sitting at (e.g.) +1 GalNAc
# - 0.2 Da is overwhelmingly more likely to be a real
# companion than a coincidental noise spike, even if the
# spectrum's 25th-percentile x 2 noise floor at that bin
# would normally reject it. The 0.3 Da bound is below the
# ``companion_tol_da`` (0.75 Da) so it triggers only when the
# peak is unambiguously at the right offset.
COMPANION_TIGHT_TOL_DA: Final[float] = 0.3

# Minimum number of peaks in a bin before we trust the bin's percentile;
# below this we fall back to the global statistic.
MIN_PEAKS_PER_BIN: Final[int] = 20

# Floor percentile: a peak is "real" only if its intensity is at or above
# this percentile of peak intensities in its m/z bin. 25th percentile is
# a conservative-but-not-punishing choice: in a real spectrum with many
# noise peaks, the 25th percentile sits above the noise but below real
# signals. The 2.0x multiplier is the safety margin -- a peak must be at
# least 2x the bin's 25th percentile to count as "above noise".
# (Originally the multiplier was applied to the *fallback* global
# statistic too; that proved too aggressive for sparse synthetic test
# spectra where every peak is signal, so the multiplier now applies
# only to the per-bin path.)
NOISE_PERCENTILE: Final[float] = 0.25
NOISE_MULTIPLIER: Final[float] = 1.8

# Envelope thresholds. M+0 is the dominant peak; the M+1 satellite is
# typically 5-20% as intense as M+0, M+2 is 1-3%, and M+3 is < 0.5%
# depending on the C-count. The previous design used a fraction of M+0
# (ENVELOPE_OK_RELATIVE) as the satellite gate, but this rejected
# every real M+2/M+3 because they're naturally small. The new gate is
# the local noise floor at the candidate's m/z -- a satellite is real
# if it rises above the local noise. This constant is retained for
# API/back-compat but is no longer used internally.
ENVELOPE_OK_RELATIVE: Final[float] = 0.10  # noqa: F841 -- retained for back-compat

# Relative satellite floor: a 13C satellite is considered "observed"
# if it clears the absolute noise floor OR a minimum fraction of M+0
# intensity. The relative floor matters most for low-intensity
# candidates where the absolute noise over-penalises the naturally
# weak M+2 (typically 4-15% of M+0 for 30-60 C glycans).
ENVELOPE_REL_M1_FRAC: Final[float] = 0.20  # M+1 expected ~44% for 40C
ENVELOPE_REL_M2_FRAC: Final[float] = 0.04  # M+2 expected ~10% for 40C
ENVELOPE_REL_M3_FRAC: Final[float] = 0.005  # M+3 expected ~1.5% for 40C

#: Confidence-interval margin for the absolute satellite noise gate.
#: A real 13C satellite must rise above the noise by enough sigma
#: that a coincidental noise spike at the same m/z is unlikely. The
#: Poisson noise model says the std-dev of a peak of expected
#: intensity ``noise`` is ``sqrt(noise)``; multiplying by this
#: factor and adding to the noise floor gives the "this peak is
#: unlikely to be a coincidence" threshold. 2.0 corresponds to
#: roughly 95% confidence. At LOW M+0 S/N (the case the user
#: complained about -- e.g. M+0=447, noise=350, M+1 expected ~197
#: = below noise), the CI gate is what protects against a
#: coincidental noise spike landing near the M+1 position and
#: being accepted as a real satellite. At HIGH M+0 S/N the CI
#: gate is dominated by noise (sqrt(noise) << noise) so it
#: effectively no-ops and the natural satellite passes.
ENVELOPE_SIGMA_FACTOR: Final[float] = 2.0

# Strict-envelope m/z window. Candidates whose m/z falls in this range
# must have M+1 AND M+2 AND M+3 all observed above the noise floor to
# count as envelope_ok=True. Outside this range (especially at high
# m/z where M+2 / M+3 fall below 1e-3 relative abundance for the
# typical 100-200 carbon glycans), only M+0 + M+1 is required and
# the label says so. The lower bound (1000) matches the user-facing
# m/z slider's minimum; the upper bound (2400) covers the "small
# glycan" regime where the molecule has ~10-30 carbons and M+2/M+3
# are still well above the noise floor.
# Strict envelope window: m/z <= 3800. The user tunes the
# strict-envelope check to this range because the small-glycan
# "M+0 + M+1 + M+2 + M+3" requirement is meaningful for any
# molecule that has few enough carbons that all three satellites
# are physically observable above the noise floor. The upper
# bound (3800) covers the regime where M+2 / M+3 still rise
# above the noise on a clean spectrum for typical
# GalNAc/Gal compositions. Above 3800, the molecule is large
# enough that M+2 / M+3 drift below the noise floor even on
# a clean spectrum, so the strict check would fail
# spuriously. The lower bound tracks the user's m/z slider
# minimum (1000) -- the strict rule applies throughout the
# user's normal operating range.
STRICT_ENVELOPE_MZ_LO: Final[float] = 1000.0
STRICT_ENVELOPE_MZ_HI: Final[float] = 3800.0

# S/N thresholds for the strict window. M+0 intensity / noise_at_target
# determines which satellite set is required:
#   >= 5x noise: HIGH quality -- require M+0+M+1+M+2+M+3
#   >= 1.5x noise: MEDIUM quality -- require M+0+M+1+M+2 (M+3 dropped,
#                  expected to be near noise at this S/N)
#   <  1.5x noise: LOW quality -- require M+0+M+1 only
STRICT_ENVELOPE_HIGH_SN_NOISE: Final[float] = 5.0
STRICT_ENVELOPE_LOW_SN_NOISE: Final[float] = 1.09

#: Minimum M+0 S/N for a candidate to be considered for any tier
#: at all. Candidates whose M+0 intensity is below this multiple of
#: the local noise floor are PURGEd: the peak itself is too close to
#: noise for any companion / series / envelope signal to be trusted.
#: The "Hide if M+0 below 1.5x noise" checkbox in app.py (default
#: ON) drops these rows from the graph and the table.
MIN_M0_SNR_FOR_KEEP: Final[float] = STRICT_ENVELOPE_LOW_SN_NOISE


#: Dalton window for the "two good candidates within X of each other
#: -- keep the lower m/z" rule. After tiering, any candidate whose
#: observed m/z is within this many Daltons of an earlier (lower-m/z)
#: non-PURGE candidate is dropped in favour of the lower one. PURGE
#: rows are skipped (they've already been filtered). The 2.2 Da bound
#: is wider than :data:`DEFAULT_NOISE_BIN_WIDTH` / 2 so two peaks in
#: different noise bins but adjacent on the m/z axis are still
#: collapsed. It is intentionally narrower than a single Gal
#: residue (162 Da) so unrelated compositions are never collapsed
#: into each other.
CANDIDATE_MZ_DEDUP_DA: Final[float] = 2.2

#: Dalton window for the "GalNAc-dominant candidate's M+k envelope
#: shadows a nearby candidate" rule. When a GalNAc-dominant
#: candidate (``n_galnac > n_gal``) sits within this many Daltons
#: of a higher-m/z candidate AND any of the first's M+1 / M+2 / M+3
#: isotope positions falls within ``da_tol`` of the second's M+0 or
#: M+1 position, the second candidate cannot be distinguished from
#: the first's isotope satellites -- the apparent M+0/M+1 of the
#: second may just be M+1/M+2 of the first. We erase (PURGE) the
#: second in that case. The 6-Da bound covers M+0..M+3 of a typical
#: glycan (3 * 1.003355 ~= 3.0 Da) plus the user's Da tolerance on
#: each side; a wider window would let unrelated compositions
#: interfere, a narrower window would miss the M+2/M+3 cases.
GALNAC_OVERLAP_ERASE_DA: Final[float] = 6.0


# --- Adduct parsing --------------------------------------------------------

_ION_TO_ADDUCT: Final[dict[str, Adduct]] = {
    "H+": Adduct.H,
    "Na+": Adduct.NA,
    "K+": Adduct.K,
}


def _adduct_from_ion(ion: str) -> Adduct:
    """Map the string label in the candidate DataFrame back to an Adduct."""
    if ion not in _ION_TO_ADDUCT:
        raise ValueError(f"unknown ion label {ion!r}; expected one of {list(_ION_TO_ADDUCT)}")
    return _ION_TO_ADDUCT[ion]


# --- Composition -> formula (closed form) ----------------------------------


def _composition_formula(n_galnac: int, n_gal: int) -> str:
    """Return the molecular formula for ``n_galnac`` GalNAc + ``n_gal`` Gal + H2O.

    GalNAc residue: C8H13NO5. Gal residue: C6H10O5. Free reducing-end water.
    Closed form:
        C = 8n + 6m
        H = 13n + 10m + 2
        N = n
        O = 5n + 5m + 1
    """
    if n_galnac < 0 or n_gal < 0:
        raise ValueError(f"residue counts must be non-negative; got n={n_galnac}, m={n_gal}")
    c = 8 * n_galnac + 6 * n_gal
    h = 13 * n_galnac + 10 * n_gal + 2
    n = n_galnac
    o = 5 * n_galnac + 5 * n_gal + 1
    return f"C{c}H{h}N{n}O{o}"


# --- Noise floor (windowed, robust) ----------------------------------------


def _bin_lo(mz: float) -> float:
    """Low edge of the noise-floor bin containing ``mz``.

    The spectrum uses two bin widths: 300 Da below
    :data:`NOISE_BIN_SPLIT_MZ`, 150 Da at or above it. The split
    keeps the per-bin noise statistic robust (>= 20 peaks per bin
    for the 25th percentile to be meaningful) at low m/z where
    peaks are dense, and gives finer-grained estimates at high m/z
    where glycan peaks are sparse and adjacent compositions sit
    ~150-200 Da apart.
    """
    if mz < NOISE_BIN_SPLIT_MZ:
        return (mz // DEFAULT_NOISE_BIN_WIDTH) * DEFAULT_NOISE_BIN_WIDTH
    offset = mz - NOISE_BIN_SPLIT_MZ
    return NOISE_BIN_SPLIT_MZ + (offset // NOISE_BIN_WIDTH_HIGH) * NOISE_BIN_WIDTH_HIGH


def _bin_width_for_lo(bin_lo: float) -> float:
    """Width of the bin whose low edge is ``bin_lo``."""
    if bin_lo < NOISE_BIN_SPLIT_MZ:
        return DEFAULT_NOISE_BIN_WIDTH
    return NOISE_BIN_WIDTH_HIGH


def _bin_index(mz: float) -> int:
    """Integer bin coordinate. Peaks with the same index land in the same bin.

    The coordinate is the bin's low edge in Da -- not a sequential
    counter -- so it is stable across the binning transition at
    :data:`NOISE_BIN_SPLIT_MZ` and so two peaks in different regions
    that happen to share a low edge still land in different dict
    entries.
    """
    return int(_bin_lo(mz))


def _noise_floor_for_mz(
    peaks: Sequence[Peak],
    target_mz: float,
) -> float:
    """Robust per-m/z-window noise floor.

    Bins ``peaks`` by m/z into the two-tier binning scheme (300 Da
    below :data:`NOISE_BIN_SPLIT_MZ`, 150 Da above) and returns
    the :data:`NOISE_PERCENTILE` of intensities in the bin that contains
    ``target_mz``, multiplied by :data:`NOISE_MULTIPLIER`. Falls back to
    the global minimum positive intensity (NOT multiplied) if the bin
    has fewer than :data:`MIN_PEAKS_PER_BIN` peaks. The fallback is
    intentionally lenient: a sparse bin means we don't trust the local
    statistic, so we accept any peak with positive intensity rather than
    rejecting the whole spectrum.

    Returns a positive float. If the input is empty or all intensities
    are zero, returns a small positive sentinel (1.0) so the caller
    never gets a zero threshold and rejects every peak.

    NOTE: this function has O(N) cost in the size of ``peaks`` because
    it iterates to build the bin. Callers that need to call it for many
    target_mz values should use :func:`_build_noise_floor_cache` to
    precompute the per-bin floors and look them up in O(1).
    """
    if not peaks:
        return 1.0

    bin_lo = _bin_lo(target_mz)
    bin_hi = bin_lo + _bin_width_for_lo(bin_lo)
    in_bin = [p.intensity for p in peaks if bin_lo <= p.mz < bin_hi and p.intensity > 0]

    if len(in_bin) >= MIN_PEAKS_PER_BIN:
        in_bin.sort()
        idx = max(0, int(round(NOISE_PERCENTILE * (len(in_bin) - 1))))
        floor = in_bin[idx] * NOISE_MULTIPLIER
        if floor > 0:
            return floor

    # Fallback: global minimum positive intensity, no multiplier. We
    # accept any non-zero peak rather than over-penalising sparse data.
    all_pos = [p.intensity for p in peaks if p.intensity > 0]
    if not all_pos:
        return 1.0
    return min(all_pos)


def _build_noise_floor_cache(
    peaks: Sequence[Peak],
) -> tuple[dict[int, float], float]:
    """Precompute the noise floor for every populated bin.

    Returns a 2-tuple ``(cache_dict, fallback_intensity)`` so the
    per-candidate inner loop can look up the floor in O(1) instead
    of doing an O(N) bin scan per call. This is the difference
    between a sub-second render and a multi-second one on a 10k-peak
    spectrum with thousands of candidates.

    Binning uses the two-tier scheme (300 Da below
    :data:`NOISE_BIN_SPLIT_MZ`, 150 Da above) so high-m/z candidates
    get finer-grained local estimates where glycan peaks are sparser.

    ``cache_dict`` maps ``bin_lo -> floor`` where the floor is
    either the bin's 25th-percentile x :data:`NOISE_MULTIPLIER` (when
    the bin has at least :data:`MIN_PEAKS_PER_BIN` peaks) or the
    global fallback intensity (when the bin is too sparse to trust
    the per-bin statistic).

    ``fallback_intensity`` is the global minimum positive intensity
    across the whole spectrum -- the value the per-call path
    :func:`_noise_floor_for_mz` returns for a sparse bin. Returning
    it as part of the cache tuple means a missing-bin lookup in
    :func:`_noise_floor_cached` is exactly consistent with the
    per-call path's behaviour, instead of returning ``min(cache.values())``
    which can disagree when the cache is sparse or empty.
    """
    bins: dict[int, list[float]] = {}
    all_pos: list[float] = []
    for p in peaks:
        if p.intensity <= 0:
            continue
        idx = _bin_index(p.mz)
        bins.setdefault(idx, []).append(p.intensity)
        all_pos.append(p.intensity)

    fallback: float = min(all_pos) if all_pos else 1.0

    out: dict[int, float] = {}
    for idx, intensities in bins.items():
        if len(intensities) >= MIN_PEAKS_PER_BIN:
            intensities.sort()
            rank = max(0, int(round(NOISE_PERCENTILE * (len(intensities) - 1))))
            floor = intensities[rank] * NOISE_MULTIPLIER
            out[idx] = floor if floor > 0 else fallback
        else:
            out[idx] = fallback
    return out, fallback


def _noise_floor_cached(
    cache: dict[int, float],
    target_mz: float,
    *,
    fallback: float | None = None,
) -> float:
    """O(1) lookup of the noise floor at ``target_mz`` from a pre-built cache.

    The cache is the first element of the 2-tuple returned by
    :func:`_build_noise_floor_cache`. ``fallback`` is the second
    element -- the global minimum positive intensity -- and is
    returned for missing bins so the lookup exactly matches what the
    per-call :func:`_noise_floor_for_mz` would compute.

    For back-compat, when ``fallback`` is omitted (e.g. older test
    fixtures passing a raw dict), the lookup returns ``1.0`` for
    empty / missing bins instead of the previous
    ``min(cache.values())`` heuristic. ``1.0`` matches the empty-
    spectrum sentinel returned by :func:`_noise_floor_for_mz`.
    """
    idx = _bin_index(target_mz)
    if idx in cache:
        return cache[idx]
    if fallback is not None:
        return fallback
    # Back-compat path: no fallback supplied. Match the per-call
    # ``_noise_floor_for_mz`` empty-input sentinel rather than the
    # previous (incorrect) ``min(cache.values())`` heuristic.
    return 1.0


# --- Sorted-by-mz peak index for binary search -----------------------------


def _sorted_mz_index(peaks: Sequence[Peak]) -> tuple[list[float], list[float]]:
    """Return (sorted mz list, sorted intensity list) for binary search."""
    paired = sorted(((p.mz, p.intensity) for p in peaks), key=lambda x: x[0])
    return [mz for mz, _ in paired], [i for _, i in paired]


def _has_peak_near(
    sorted_mz: list[float],
    sorted_intensity: list[float],
    target_mz: float,
    da_tol: float,
    min_intensity: float,
) -> bool:
    """Return True iff a peak is within ``da_tol`` of ``target_mz`` and at/above ``min_intensity``."""
    if not sorted_mz:
        return False
    lo = target_mz - da_tol
    hi = target_mz + da_tol
    # bisect_right gives the first index with mz > hi; bisect_left gives
    # the first with mz >= lo.
    start = bisect.bisect_left(sorted_mz, lo)
    end = bisect.bisect_right(sorted_mz, hi)
    for i in range(start, end):
        if sorted_intensity[i] >= min_intensity:
            return True
    return False


def _strongest_peak_near(
    sorted_mz: list[float],
    sorted_intensity: list[float],
    target_mz: float,
    da_tol: float,
) -> tuple[float, float]:
    """Strongest peak within ``da_tol`` of ``target_mz`` and its m/z.

    Returns ``(0.0, 0.0)`` when no peak is in the window. Used by the
    screener to determine the M+0 intensity of a candidate: the
    upstream solver put a peak within ``da_tol`` of the candidate's
    m/z, and the strongest in that window is the one that the
    screener evaluates against. Re-deriving it here keeps the
    caller decoupled from ``_score_envelope``'s internal logic.
    """
    if not sorted_mz:
        return 0.0, 0.0
    lo = target_mz - da_tol
    hi = target_mz + da_tol
    start = bisect.bisect_left(sorted_mz, lo)
    end = bisect.bisect_right(sorted_mz, hi)
    best_intensity = 0.0
    best_mz = 0.0
    for i in range(start, end):
        if sorted_intensity[i] > best_intensity:
            best_intensity = sorted_intensity[i]
            best_mz = sorted_mz[i]
    return best_intensity, best_mz


# --- 13C envelope ----------------------------------------------------------

# Monoisotopic atomic masses (CIAAW). Only the elements present in our
# composition formula (C, H, N, O) are needed; the adduct (H+/Na+/K+)
# is added separately as a single shift.
_ATOMIC_MASS: Final[dict[str, float]] = {
    "C": 12.0000000,
    "H": 1.0078250319,
    "N": 14.0030740052,
    "O": 15.9949146221,
}

# Natural abundance of 13C (the only non-monoisotopic isotope that
# contributes meaningfully to the M+0..M+3 envelope in the m/z range
# of interest -- 13N and 17O are <0.04% and their contributions are
# below the screener's intensity gate).
_C13_ABUNDANCE: Final[float] = 0.0111

# Mass difference between 13C and 12C, in Da. M+k is the monoisotopic
# mass plus k of these shifts.
_C13_MASS_SHIFT: Final[float] = 1.003355

# Process-level cache for the theoretical envelope keyed by
# (n_galnac, n_gal, adduct_mass). Many candidates share the same
# composition + adduct (e.g. multiple ppm-close matches for the same
# m/z), so caching avoids re-running the multinomial expansion for
# each one. Bounded size so a long-running session doesn't grow
# unbounded if the user uploads a file with hundreds of distinct
# compositions.
_ENVELOPE_CACHE: dict[tuple[int, int, float], list[tuple[float, float]] | None] = {}
_ENVELOPE_CACHE_MAX: int = 1024


def _neutral_formula_counts(n_galnac: int, n_gal: int) -> dict[str, int]:
    """Return the closed-form element counts for ``n_galnac`` GalNAc + ``n_gal`` Gal + H2O.

    The values are guaranteed to be non-negative for any non-negative
    input (H is at least 2, O is at least 1, C is zero only at n=0,m=0,
    N is zero only at n=0). Callers needing a mass should sum
    ``count * _ATOMIC_MASS[element]``.
    """
    if n_galnac < 0 or n_gal < 0:
        raise ValueError(
            f"residue counts must be non-negative; got n={n_galnac}, m={n_gal}"
        )
    return {
        "C": 8 * n_galnac + 6 * n_gal,
        "H": 13 * n_galnac + 10 * n_gal + 2,
        "N": n_galnac,
        "O": 5 * n_galnac + 5 * n_gal + 1,
    }


def _monoisotopic_mass_neutral(n_galnac: int, n_gal: int) -> float:
    """Monoisotopic mass of the neutral composition, in Daltons.

    Replaces the previous dependence on :mod:`molmass` for the
    monoisotopic mass, which the envelope helper also needed. The
    closed-form sum uses the same atomic masses molmass uses
    internally; differences are at the 1e-4 Da level and below the
    screener's tolerance.
    """
    counts = _neutral_formula_counts(n_galnac, n_gal)
    return sum(_ATOMIC_MASS[elem] * count for elem, count in counts.items())


def _c13_envelope(
    n_galnac: int, n_gal: int, adduct_mass: float
) -> list[tuple[float, float]]:
    """Closed-form 13C isotopologue envelope for the composition + adduct.

    Returns a list of ``(mz, intensity_pct)`` tuples starting at M+0
    and continuing through the heaviest isotopologue whose relative
    intensity is at least 0.05% of the M+0 peak. Intensities are
    normalised so M+0 = 100.

    The expansion is a binomial on the carbon isotope distribution:
    for ``C_count`` carbon atoms at 13C abundance ``p = 0.0111``, the
    probability of exactly ``k`` atoms being 13C is
    ``C(C_count, k) * p**k * (1-p)**(C_count - k)``. M+0 is the
    monoisotopic mass; M+k adds ``k * 1.003355`` Da.
    """
    counts = _neutral_formula_counts(n_galnac, n_gal)
    c_count = counts["C"]
    m0 = _monoisotopic_mass_neutral(n_galnac, n_gal) + adduct_mass

    # k = 0, 1, 2, ... up to the point where probability drops below
    # the cutoff. For C_count = 100 (large glycan) the mean M+k is
    # ~1.1, so we typically need only k = 0..5 to cover the screener's
    # M+0..M+3 requirement with a generous tail.
    k = np.arange(c_count + 1, dtype=np.float64)
    p = _C13_ABUNDANCE
    # log-space binomial to avoid overflow at large C_count.
    log_binom = _log_binomial_coefficient(c_count, k)
    log_prob = log_binom + k * np.log(p) + (c_count - k) * np.log1p(-p)
    prob = np.exp(log_prob)
    prob = prob / prob[0] * 100.0  # normalise so M+0 = 100
    keep = prob >= 0.05
    mz = m0 + np.arange(c_count + 1, dtype=np.float64) * _C13_MASS_SHIFT
    return [(float(mz[i]), float(prob[i])) for i in range(c_count + 1) if keep[i]]


# Cache log-factorials since the same composition may be requested many
# times. The largest n we see is the carbon count in a (24, 30)
# composition, ~372; the cache tops out around 400 entries.
_LOG_FACTORIAL_CACHE: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.6931471805599453}


def _log_factorial(n: int) -> float:
    """``log(n!)`` with a tiny cache. Stdlib-only; n is bounded by ~400."""
    cached = _LOG_FACTORIAL_CACHE.get(n)
    if cached is not None:
        return cached
    # Fill the cache for all missing values up to n.
    start = max(_LOG_FACTORIAL_CACHE.keys())
    acc = _LOG_FACTORIAL_CACHE[start]
    for k in range(start + 1, n + 1):
        acc += math.log(k)
        _LOG_FACTORIAL_CACHE[k] = acc
    return _LOG_FACTORIAL_CACHE[n]


def _log_binomial_coefficient(n: int, k: np.ndarray) -> np.ndarray:
    """Element-wise ``log(C(n, k))`` for scalar ``n`` and array ``k``.

    Computed as ``log(n!) - log(k!) - log((n-k)!)`` via :func:`_log_factorial`.
    No scipy or numpy special-functions dependency.
    """
    n_f = _log_factorial(n)
    # Populate the cache up to n once, then build a vectorised lookup
    # by reading the small dict.
    lf_k = np.empty(k.shape, dtype=np.float64)
    lf_nk = np.empty(k.shape, dtype=np.float64)
    for idx in np.ndindex(k.shape):
        ki = int(k[idx])
        nki = n - ki
        # _log_factorial populates the cache; the second call is O(1).
        lf_k[idx] = _log_factorial(ki)
        lf_nk[idx] = _log_factorial(nki)
    return n_f - lf_k - lf_nk


def _envelope_for(
    n_galnac: int,
    n_gal: int,
    adduct_mass: float,
) -> list[tuple[float, float]] | None:
    """Theoretical isotopologue envelope, as ``[(mz, intensity_pct), ...]``.

    Returns the neutral envelope, shifted by the adduct mass (Na+ and
    K+ are monoisotopic, so the shift is a single value). Returns
    ``None`` only if ``_c13_envelope`` somehow fails for an unparseable
    composition, which should not occur for the closed-form formulas
    the screener generates.

    Results are memoised in a process-level dict because the same
    composition + adduct often appears in many candidate rows.
    """
    key = (n_galnac, n_gal, adduct_mass)
    if key in _ENVELOPE_CACHE:
        return _ENVELOPE_CACHE[key]
    # Evict the oldest entry if we're at the cap. dicts preserve
    # insertion order in CPython 3.7+, so popitem() removes the oldest.
    if len(_ENVELOPE_CACHE) >= _ENVELOPE_CACHE_MAX:
        _ENVELOPE_CACHE.popitem()

    try:
        env = _c13_envelope(n_galnac, n_gal, adduct_mass)
    except (ValueError, KeyError, TypeError):
        _ENVELOPE_CACHE[key] = None
        return None
    _ENVELOPE_CACHE[key] = env
    return env


def _score_envelope(
    n_galnac: int,
    n_gal: int,
    adduct: Adduct,
    sorted_mz: list[float],
    sorted_intensity: list[float],
    da_tol: float,
    target_mz: float,
    noise_at_target: float = 0.0,
) -> tuple[bool, str, bool]:
    """Check whether the candidate's theoretical isotope envelope is observed.

    ``da_tol`` is an absolute Dalton tolerance that the user controls
    via the sidebar slider. The M+0 search uses ``da_tol`` directly
    (no ppm scaling) and satellite peaks are searched with the same
    tolerance so the whole envelope check is consistent.

    Returns (envelope_ok, note, envelope_skipped).
    ``envelope_skipped`` is True when the envelope could not be
    evaluated (molmass unavailable, no peaks at the candidate's m/z).
    In that case the result is "we don't know" -- companion + series
    can still rescue the candidate to YELLOW. An *active* failure
    (envelope_ok=False, envelope_skipped=False) is a hard negative
    signal and cannot be rescued.
    """
    env = _envelope_for(n_galnac, n_gal, adduct.mass)
    if env is None:
        return False, "isotope envelope not checkable", True
    if not env:
        return False, "", True

    # Anchor the envelope on the OBSERVED candidate m/z, not the
    # theoretical M+0. The upstream composition solver matched the
    # candidate to within ``da_tol`` (or the ppm tolerance) of the
    # theoretical m/z, so the candidate's m/z may differ from
    # ``env[0][0]`` by up to ~0.7 Da. If we search for M+0..M+3
    # against the theoretical positions and then check the band
    # rule (0.75-1.2 Da from previous), an off-by-0.5-Da candidate
    # would have every satellite land outside the band because the
    # whole envelope is shifted by the same offset. The fix is to
    # treat the candidate's m/z as M+0 and look for satellites at
    # observed_mz + k * 1.003355.
    m0_observed = target_mz

    # M+0 search: the candidate's m/z IS the M+0 by construction
    # (the upstream solver just put a peak within da_tol of this
    # position). The strongest peak in the local window is the
    # matched M+0.
    lo = m0_observed - da_tol
    hi = m0_observed + da_tol
    start = bisect.bisect_left(sorted_mz, lo)
    end = bisect.bisect_right(sorted_mz, hi)
    matched_intensity = 0.0
    matched_index = -1
    for i in range(start, end):
        if sorted_intensity[i] > matched_intensity:
            matched_intensity = sorted_intensity[i]
            matched_index = i

    if matched_intensity <= 0:
        return False, "no observed peaks at theoretical isotope positions", True

    # The "above-noise" gate for satellite peaks. M+1 / M+2 / M+3 are
    # naturally much smaller than M+0 (typically 5-20% of M+0 for M+1,
    # 1-3% for M+2, 0.1% for M+3 in a typical 20-60 carbon glycan), so
    # using a fraction of M+0 as the gate (the old behaviour) would
    # reject every real M+2/M+3 just because the math says they're
    # small. The physically correct gate is the local noise floor: a
    # satellite is real if it rises above the noise at its m/z. The
    # caller passes ``noise_at_target`` (the M+0 bin's floor); we
    # reuse it as a conservative lower bound for all satellites.
    # Hybrid absolute-OR-relative threshold. A satellite is real if
    # it clears the absolute noise floor OR is a meaningful fraction
    # of M+0. Without the relative floor, low-intensity candidates
    # (small M+0) get their naturally-weak M+2 killed by the
    # absolute noise (typical 25th-percentile x 2.0). The two
    # thresholds are OR'd in the satellite loop below, not AND'd.
    m0_intensity = float(matched_intensity)
    rel_floor_m1 = ENVELOPE_REL_M1_FRAC * m0_intensity
    rel_floor_m2 = ENVELOPE_REL_M2_FRAC * m0_intensity
    rel_floor_m3 = ENVELOPE_REL_M3_FRAC * m0_intensity

    # Poisson confidence-interval absolute gate. A coincidental noise
    # spike at the satellite position has expected intensity
    # ``noise_at_target`` and std-dev ``sqrt(noise_at_target)``
    # (Poisson). For the satellite to be unlikely a noise spike, it
    # must clear ``noise + sigma * sqrt(noise)``. The 2.0-sigma bound
    # corresponds to ~95% confidence. This is what catches the
    # 1462-Da, M+0=447, noise=350 case the user complained about:
    # with the old pure-noise gate, a noise spike at 360 was
    # accepted as a real M+1 even though the real M+1 (~197) is
    # BELOW noise and therefore indistinguishable. With the CI gate
    # the abs threshold becomes 350 + 2*sqrt(350) ~= 387, so the
    # 360 noise spike is rejected. At HIGH M+0 S/N, sqrt(noise) is
    # much smaller than noise, so the CI gate is effectively the
    # noise floor (no-op) and naturally-strong satellites pass as
    # before.
    ci_abs_gate = noise_at_target + ENVELOPE_SIGMA_FACTOR * math.sqrt(max(noise_at_target, 0.0))

    # Disable the relative gate at LOW M+0 S/N. The relative gate
    # is physically meaningful only when M+0 is confidently above
    # noise -- if M+0 is itself barely above the noise floor, the
    # "M+0 is real" assumption is shaky and the "satellite is X% of
    # M+0" prediction is not trustworthy. We gate the relative
    # floor off entirely below the MEDIUM bar (1.5x noise) -- at
    # that S/N the user has already decided M+0 is real enough to
    # require M+2 as evidence (MEDIUM bucket), and the natural
    # satellite fractions are large enough to be meaningful.
    # Below MEDIUM, the satellite is judged purely by the CI abs
    # gate, and coincidental noise spikes are rejected.
    _sn_ratio = (
        m0_intensity / noise_at_target if noise_at_target > 0 else float("inf")
    )
    rel_gate_enabled = _sn_ratio >= 1.5

    # Did we observe M+1 / M+2 / M+3 above the local noise floor?
    # molmass returns isotopologues out to ~20+ entries (with
    # predicted abundances below 1e-10 by the tail). Without a cap,
    # the search would match every real peak in the spectrum that
    # happens to land near an N*1.003-Da offset, producing bogus
    # "M+21" / "M+22" / "M+23" hits. The physically meaningful 13C
    # isotope peaks are the first 3 satellites -- M+1, M+2, M+3 --
    # and the label below reports exactly that range.
    observed_isotopes: list[int] = []  # [+1, +2, +3] of those seen
    # The OBSERVED M+0 m/z: the actual peak the M+0 search landed
    # on. Subsequent satellites chain from THIS position, not from
    # the theoretical. If the candidate was matched 0.5 Da off the
    # theoretical (well inside the user's Da tolerance), the M+0
    # peak is at target_mz + 0.5 and the M+1 peak in the spectrum
    # is at M+0 + 1.003 -- anchoring on the actual M+0 keeps the
    # band test in 0.75-1.2 Da range. Anchoring on the theoretical
    # M+0 (= target_mz) would put the M+1 band test at
    # target_mz + 1.5 (gap 1.5) -- outside the band, which is the
    # bug we just fixed.
    if matched_index >= 0:
        prev_observed_mz = float(sorted_mz[matched_index])
    else:
        prev_observed_mz = m0_observed
    for isotope_idx in (1, 2, 3):
        # M+k is searched at the previous observed isotope's m/z +
        # 1.003355 Da. Each step chains from the previous one.
        env_mz = prev_observed_mz + _C13_MASS_SHIFT
        # Isotope band rule: each M+k must be 0.75 to 1.2 Da above
        # the previous isotope. The 13C shift is 1.003355 Da, so
        # this band is a tight (0.75 / 1.2 of d) relative-to-prev
        # check identical in spirit to the series-ladder gap test.
        # A peak closer than 0.75 Da is a different species (not
        # 13C); a peak further than 1.2 Da is not a single 13C
        # substitution (could be 13C + something else, or
        # unrelated). In either case we do not count it as an
        # observed satellite.
        lo = env_mz - da_tol
        hi = env_mz + da_tol
        start = bisect.bisect_left(sorted_mz, lo)
        end = bisect.bisect_right(sorted_mz, hi)
        best_idx = -1
        best_intensity = 0.0
        for i in range(start, end):
            if sorted_intensity[i] > best_intensity:
                best_intensity = sorted_intensity[i]
                best_idx = i
        if best_idx < 0:
            continue
        peak_mz = sorted_mz[best_idx]
        # Band: peak must be 0.75 to 1.2 Da above the previous
        # observed isotope's m/z.
        gap = peak_mz - prev_observed_mz
        if gap < ISOTOPE_GAP_LO or gap > ISOTOPE_GAP_HI:
            continue
        # True OR of absolute noise gate and relative M+0 fraction.
        # A satellite is real if it clears EITHER. The 1.0 floor on
        # the relative side prevents zero-intensity noise from
        # squeaking through. See the threshold comment above.
        if isotope_idx == 1:
            rel_floor = rel_floor_m1
        elif isotope_idx == 2:
            rel_floor = rel_floor_m2
        else:
            rel_floor = rel_floor_m3
        if best_intensity >= ci_abs_gate or (
            rel_gate_enabled and best_intensity >= rel_floor
        ):
            observed_isotopes.append(isotope_idx)
            # Chain: the next isotope anchors from THIS peak.
            prev_observed_mz = peak_mz
        else:
            # The peak is in the band (passed gap test) but did not
            # clear either threshold. Advance the chain anyway: a
            # weak-but-real M+2 at this m/z is a stronger chain
            # anchor than the previous observed isotope, and without
            # advancing, the NEXT satellite search (M+k+1) would
            # re-match this same position as if it were M+k+1 -- a
            # false positive when the M+k+1 relative floor is low
            # (e.g. M+3 floor is 0.5% of M+0).
            prev_observed_mz = peak_mz

    m0_ok = matched_index >= 0

    # Strict envelope: for candidates in the small-molecule window
    # (1000-3800 Da, ~10-50 carbons), the user requires M+1 AND M+2
    # to be observed. Outside this window -- especially at high m/z
    # where M+2 / M+3 fall below ~0.1% relative abundance for the
    # typical 100-200 carbon glycans -- only M+0 + M+1 is required
    # (the more permissive "any satellite counts" rule). The decision
    # is signalled in the note so the user can see why a candidate
    # was promoted or held back.
    # M+3 is OPTIONAL (typically ~1-3% of M+0, often below noise). The
    # new strict requirement is M+0 + M+1 + M+2 -- M+2 at ~4-15% of M+0
    # is well within the linear range of a normal mass spec.
    # Required satellites scale with the M+0 signal-to-noise ratio.
    # A high-quality candidate (M+0 well above noise) should have
    # all four satellites visible -- M+3 is naturally ~1-3% of M+0
    # for 40-50 C glycans, well above the noise floor when M+0 is
    # 5x or more above noise. A marginal candidate (M+0 near the
    # noise floor) cannot be expected to show M+3; M+2 is the
    # practical lower bar for these. The S/N scaling applies in
    # BOTH the strict window AND outside it -- M+2 is observable
    # for any glycan (large or small) at the same ~4-15% of M+0
    # abundance, so requiring M+2 outside the strict window too
    # catches the same incomplete-envelope problem the strict
    # window was meant to catch, just with a wider m/z net.
    if m0_intensity >= STRICT_ENVELOPE_HIGH_SN_NOISE * noise_at_target:
        required_satellites = {1, 2, 3}
    elif m0_intensity >= STRICT_ENVELOPE_LOW_SN_NOISE * noise_at_target:
        required_satellites = {1, 2}
    else:
        required_satellites = {1}
    # When GalNAc outnumbers Gal (n_galnac > n_gal), the user's rule
    # is that M+1, M+2, AND M+3 must all clear noise -- the "HIGH"
    # bar, regardless of S/N. The biology rationale: a candidate
    # with more GalNAc than Gal is structurally unusual (GalNAc
    # should be capped by the Gal backbone on O-glycans), so a
    # incomplete isotope envelope is more likely to be a coincidental
    # match and we want the full 4-satellite evidence before we
    # promote it.
    if n_galnac > n_gal:
        required_satellites = {1, 2, 3}
    observed_set = set(observed_isotopes)
    envelope_ok = m0_ok and required_satellites.issubset(observed_set)

    if m0_ok and observed_isotopes:
        isotope_list = ", ".join(f"M+{i}" for i in observed_isotopes)
        if not envelope_ok:
            # M+0 + at least one satellite is present, but the S/N
            # bar requires more satellites than we observed. Name the
            # missing ones and report the S/N so the user can see
            # which bar was applied at this m/z.
            missing = sorted(required_satellites - observed_set)
            missing_list = ", ".join(f"M+{i}" for i in missing)
            return False, (
                f"13C envelope partial (M+0, {isotope_list}) -- "
                f"requires {missing_list} for {target_mz:.1f} m/z"
                f" (S/N={m0_intensity / max(noise_at_target, 1e-9):.1f})"
            ), False
        return True, f"13C envelope observed (M+0, {isotope_list})", False
    if m0_ok and not observed_isotopes:
        # M+0 present but no satellites at all. This is the "only
        # M+0" case and it fails the envelope check in BOTH windows
        # -- the user explicitly requires at least M+1 above the
        # local noise floor. Previously this returned envelope_ok=True
        # which was wrong outside the strict window too. This is an
        # *active failure* (envelope was fully evaluated; satellites
        # just weren't there), not a skipped evaluation.
        return False, (
            f"13C envelope requires M+1 (only M+0 present at {target_mz:.1f} m/z)"
        ), False
    if m0_ok:
        return True, "13C M+0 present, no satellite above threshold", False
    return False, "13C envelope not observed", False


# --- Companion peaks -------------------------------------------------------


def _score_companion(
    target_mz: float,
    sorted_mz: list[float],
    sorted_intensity: list[float],
    da_tol: float,
    noise_at_target: float,
    mz_max: float = float("inf"),
    noise_cache: dict[int, float] | None = None,
    companion_tol_da: float = 0.75,
    noise_fallback: float | None = None,
) -> tuple[int, list[str], bool]:
    """Count companion offset classes and return human-readable notes.

    Each offset class is checked independently. The presence of any peak
    within ``da_tol`` of ``target_mz + offset`` and above the local
    noise floor contributes 1 to the score. Combinations (pairs) are
    also checked because they are strong evidence -- e.g. +1 GalNAc
    +1 Gal is the natural step in a biosynthetic ladder.

    ``noise_at_target`` is the per-bin floor at the candidate's own m/z.
    This is used as a fallback when ``noise_cache`` is not provided.

    ``noise_cache`` is a pre-built bin-indexed dict from
    :func:`_build_noise_floor_cache`. When provided, the noise floor
    for each companion offset is evaluated at the OFFSET m/z, not at
    the candidate's m/z. This is the correct behaviour: a companion
    at +162 Da is in a different m/z region with potentially different
    background intensity, and using the candidate's noise floor can
    either falsely reject a real companion (when the offset region
    has higher noise) or falsely accept a noise spike (when the
    offset region has lower noise). With ``noise_cache`` provided,
    each offset uses its own region's floor.

    ``noise_fallback`` is the second element of the 2-tuple returned by
    :func:`_build_noise_floor_cache` -- the global minimum positive
    intensity. It is threaded into :func:`_noise_floor_cached` so a
    missing-bin lookup returns the same value the per-call
    :func:`_noise_floor_for_mz` would compute, instead of the old
    (incorrect) ``min(cache.values())`` heuristic. When ``None``,
    ``_noise_floor_cached`` falls back to the empty-input sentinel
    ``1.0`` for compatibility with older test fixtures.

    ``mz_max`` is the upper bound of the spectrum. Offsets that would
    land past the spectrum end are not counted as failures -- the
    instrument can't see what isn't acquired. The third return value
    is True if at least one offset was skipped for this reason. The
    caller uses this to decide whether "score_companion == 0" means
    "no companion in the spectrum" (negative evidence) or "all
    potential companions were off the edge" (no evidence either way,
    should not penalise the candidate).
    """
    count = 0
    notes: list[str] = []
    skipped = False
    for offset, label in _SINGLE_OFFSETS:
        candidate_mz = target_mz + offset
        if candidate_mz > mz_max:
            skipped = True
            continue
        # Per-offset noise floor: the floor at the offset m/z, not
        # at the candidate's m/z. Falls back to noise_at_target if
        # the cache wasn't provided (e.g. unit tests).
        if noise_cache is not None:
            local_noise = _noise_floor_cached(
                noise_cache, candidate_mz, fallback=noise_fallback
            )
        else:
            local_noise = noise_at_target
        # Find the strongest peak near the offset. _has_peak_near
        # is binary -- it returns True/False. We re-do the bisect
        # here so the note can quote the actual intensity and m/z
        # of the peak we considered, so the user can see WHY a
        # real-looking companion was rejected (e.g. "found at
        # 4258.5 intensity 30, local noise 100 -- below floor").
        tol_da = max(da_tol, companion_tol_da)
        lo = candidate_mz - tol_da
        hi = candidate_mz + tol_da
        start = bisect.bisect_left(sorted_mz, lo)
        end = bisect.bisect_right(sorted_mz, hi)
        # Pick the strongest in-range peak (regardless of noise).
        best_idx = -1
        best_intensity = 0.0
        for i in range(start, end):
            if sorted_intensity[i] > best_intensity:
                best_intensity = sorted_intensity[i]
                best_idx = i
        if best_idx < 0:
            # No peak at all in the search window. Check the extended
            # band (1.5x tolerance) for a near-miss peak so the user
            # can see "there's a peak 1.2 Da off the theoretical +Gal
            # position" -- useful when the real companion is just
            # outside the 0.75 Da window. We do NOT count it; we
            # just surface it in the note.
            ext_tol = tol_da * 2.0
            ext_lo = candidate_mz - ext_tol
            ext_hi = candidate_mz + ext_tol
            ext_start = bisect.bisect_left(sorted_mz, ext_lo)
            ext_end = bisect.bisect_right(sorted_mz, ext_hi)
            ext_best_idx = -1
            ext_best_intensity = 0.0
            for i in range(ext_start, ext_end):
                # Skip peaks already inside the main window.
                if lo <= sorted_mz[i] <= hi:
                    continue
                if sorted_intensity[i] > ext_best_intensity:
                    ext_best_intensity = sorted_intensity[i]
                    ext_best_idx = i
            if ext_best_idx >= 0:
                near_peak_mz = sorted_mz[ext_best_idx]
                near_diff = near_peak_mz - candidate_mz
                notes.append(
                    f"{label}: near miss at {near_peak_mz:.4f} "
                    f"({near_diff:+.2f} Da off theoretical)"
                )
            # No peak at all in the search window -- truly absent.
            continue
        peak_mz = sorted_mz[best_idx]
        if best_intensity >= local_noise:
            count += 1
            notes.append(label)
        else:
            # Peak present but below local noise. Two sub-cases:
            #   (a) The peak is within ``COMPANION_TIGHT_TOL_DA`` of
            #       the theoretical offset. In that case it's
            #       almost certainly a real companion that the
            #       noise floor is over-penalising (e.g. a
            #       +1 GalNAc peak at 0.2 Da off theoretical in a
            #       bin where the 25th-percentile x 2 floor is
            #       high). Count it as a companion and add a
            #       "tight" tag to the note so the user knows
            #       the override fired.
            #   (b) Otherwise: report the peak's actual numbers
            #       in the diagnostic note so the user knows the
            #       companion was found-but-rejected.
            peak_diff = peak_mz - candidate_mz
            if abs(peak_diff) <= COMPANION_TIGHT_TOL_DA:
                count += 1
                notes.append(f"{label} (tight, {peak_diff:+.2f} Da)")
            else:
                notes.append(
                    f"{label}: peak at {peak_mz:.4f} (int {best_intensity:.0f}, "
                    f"floor {local_noise:.0f}, {peak_diff:+.2f} Da off theoretical)"
                )
    for offset, label in _PAIR_OFFSETS:
        candidate_mz = target_mz + offset
        if candidate_mz > mz_max:
            skipped = True
            continue
        if noise_cache is not None:
            local_noise = _noise_floor_cached(
                noise_cache, candidate_mz, fallback=noise_fallback
            )
        else:
            local_noise = noise_at_target
        tol_da = max(da_tol, companion_tol_da)
        lo = candidate_mz - tol_da
        hi = candidate_mz + tol_da
        start = bisect.bisect_left(sorted_mz, lo)
        end = bisect.bisect_right(sorted_mz, hi)
        best_idx = -1
        best_intensity = 0.0
        for i in range(start, end):
            if sorted_intensity[i] > best_intensity:
                best_intensity = sorted_intensity[i]
                best_idx = i
        if best_idx < 0:
            continue
        peak_mz = sorted_mz[best_idx]
        if best_intensity >= local_noise:
            count += 1
            notes.append(label)
        else:
            # Same "tight-offset" override as the single-offset
            # loop: a peak within COMPANION_TIGHT_TOL_DA of the
            # theoretical position is almost certainly a real
            # companion even if it sits below the local noise
            # floor. Without this override a +1 GalNAc +1 Gal
            # companion at 0.2 Da off theoretical would be
            # rejected by the noise floor and reported only as a
            # diagnostic, which is misleading because the peak
            # is unambiguously at the right offset.
            peak_diff = peak_mz - candidate_mz
            if abs(peak_diff) <= COMPANION_TIGHT_TOL_DA:
                count += 1
                notes.append(f"{label} (tight, {peak_diff:+.2f} Da)")
            else:
                notes.append(
                    f"{label}: peak at {peak_mz:.4f} (int {best_intensity:.0f}, "
                    f"floor {local_noise:.0f})"
                )
    return count, notes, skipped


# --- Series / 3-in-a-row ---------------------------------------------------


def _score_series(
    target_mz: float,
    sorted_mz: list[float],
    sorted_intensity: list[float],
    da_tol: float,
    noise_cache: dict[int, float],
    mz_max: float = float("inf"),
    companion_tol_da: float = 0.75,
    noise_fallback: float | None = None,
) -> tuple[int, list[str], bool]:
    """Longest detected ladder of additional peaks at a fixed offset.

    For each residue-style offset ``d`` in :data:`SERIES_OFFSETS`, walk
    forward: X+d, X+2d, X+3d, ... requiring each step's observed peak
    to be within 0.75-1.2 of d from the previous step's position and
    above the local noise floor. Returns the maximum chain length
    (number of additional peaks after the candidate) found across all
    offsets, and a label naming the offset class.

    The ``noise_cache`` is a pre-built bin-indexed dict from
    :func:`_build_noise_floor_cache`; passing it in avoids the O(N)
    bin-scan cost per step that would otherwise dominate the runtime
    on large spectra.

    ``noise_fallback`` is the second element of the 2-tuple returned by
    :func:`_build_noise_floor_cache` -- the global minimum positive
    intensity. It is threaded into :func:`_noise_floor_cached` so a
    missing-bin lookup returns the same value the per-call
    :func:`_noise_floor_for_mz` would compute, instead of the old
    (incorrect) ``min(cache.values())`` heuristic. When ``None``,
    ``_noise_floor_cached`` falls back to the empty-input sentinel
    ``1.0`` for compatibility with older test fixtures.

    ``mz_max`` is the spectrum's upper bound. A step that would land
    past the spectrum end is treated as "data unavailable" and the
    ladder is allowed to count the steps that ARE in range, rather
    than failing the whole ladder. The third return value is True
    if at least one offset ladder was truncated by the mz_max check.
    The caller uses this to decide whether series_len=0 means "no
    ladder in the spectrum" (negative evidence) or "all ladder steps
    were off the edge" (no evidence either way).
    """
    best_len = 0
    best_label = ""
    skipped = False
    for d, label in SERIES_OFFSETS:
        prev = target_mz
        steps_found = 0
        # We require SERIES_MIN_LENGTH additional peaks. Cap the walk
        # at 20 steps to avoid pathological scans.
        for step in range(1, 21):
            expected = target_mz + d * step
            # If the expected step is past the spectrum end, stop
            # counting here -- we cannot fault the candidate for
            # data the instrument didn't acquire. Mark the offset
            # as skipped so the caller knows the ladder was
            # truncated, not failed.
            if expected > mz_max:
                skipped = True
                break
            tol_da = max(da_tol, companion_tol_da)
            lo = expected - tol_da
            hi = expected + tol_da
            start = bisect.bisect_left(sorted_mz, lo)
            end = bisect.bisect_right(sorted_mz, hi)
            # Pick the candidate peak in the window. If none, the chain
            # is broken at this step.
            found_mz: float | None = None
            local_noise = _noise_floor_cached(
                noise_cache, expected, fallback=noise_fallback
            )
            for i in range(start, end):
                if sorted_intensity[i] >= local_noise:
                    found_mz = sorted_mz[i]
                    break
            if found_mz is None:
                break
            # Gap check: the observed peak should be within
            # 0.75-1.2 of d from the previous step.
            gap = found_mz - prev
            if gap < d * SERIES_GAP_LO or gap > d * SERIES_GAP_HI:
                break
            prev = found_mz
            steps_found += 1
        if steps_found > best_len:
            best_len = steps_found
            best_label = label
    return (
        best_len,
        ([f"{best_label} x{best_len}"] if best_len >= SERIES_MIN_LENGTH and best_label else []),
        skipped,
    )


# --- Tier mapping ----------------------------------------------------------


def _tier_from_scores(
    has_companion: bool,
    series_len: int,
    envelope_ok: bool,
    tolerance_ok: bool,
    *,
    companion_skipped: bool = False,
    series_skipped: bool = False,
    envelope_skipped: bool = False,
    m0_sn_ok: bool = True,
) -> str:
    """Map (companion, series, envelope, tolerance, m0 S/N) to a tier label.

    ``companion_skipped`` / ``series_skipped`` / ``envelope_skipped`` are
    True when that check was *truncated* because the offset would land
    past the spectrum end, or because the candidate's m/z fell outside
    the strict envelope window. In that case the absence is NOT
    negative evidence -- the instrument simply didn't acquire that
    m/z, so we cannot fault the candidate for it.

    An *actively failed* envelope (envelope_ok=False, envelope_skipped=False)
    is a hard negative signal: the candidate's m/z is in a region where
    a real peak would have a visible M+1. That cannot be rescued by
    companion or series success.

    ``m0_sn_ok`` (default True) is False when the M+0 peak itself is
    below :data:`MIN_M0_SNR_FOR_KEEP` times the local noise floor.
    In that case the candidate is PURGEd regardless of any other
    check -- a peak that doesn't clear the noise is not a real
    signal, so companion / series / envelope matches are
    coincidental and not informative.
    """
    if not m0_sn_ok:
        return "PURGE"
    if not tolerance_ok:
        return "PURGE"
    if not envelope_ok:
        if envelope_skipped:
            # We couldn't evaluate the envelope (e.g. candidate m/z is
            # outside the strict envelope window, or M+0 itself wasn't
            # found at the right spot). That's neutral, not negative.
            # Defer to companion + series.
            if has_companion or series_len >= SERIES_MIN_LENGTH:
                return "YELLOW"
            if companion_skipped or series_skipped:
                return "YELLOW"
            return "RED"
        # Envelope actively failed (e.g. no M+1 found). That's a hard
        # negative signal -- the candidate's m/z is in a region where
        # a real peak would have a visible M+1. Companion + series
        # success cannot rescue it.
        return "RED"
    # Envelope OK
    if has_companion and series_len >= SERIES_MIN_LENGTH:
        return "GREEN"
    if has_companion or series_len >= SERIES_MIN_LENGTH:
        return "YELLOW"
    # No companion and no series in-range. If both were skipped
    # (off-spectrum), promote to YELLOW -- we have no negative
    # evidence we can evaluate. This is the high-m/z bias fix:
    # a 5800-Da candidate in a 5998-Da file is no longer
    # penalised for the off-spectrum companion / series offsets.
    if companion_skipped and series_skipped:
        return "YELLOW"
    if companion_skipped or series_skipped:
        # One check was evaluable and found nothing; one was off-
        # spectrum. Treat as YELLOW -- we have one positive signal
        # (envelope) and ambiguous evidence on the others.
        return "YELLOW"
    return "RED"


# --- Public entry point ----------------------------------------------------


def screen_candidates(
    df: pd.DataFrame,
    spectrum_peaks: Sequence[Peak],
    *,
    da_tol: float = 0.7,
    strictness: str = "strict",
    spectrum: Spectrum | None = None,
) -> pd.DataFrame:
    """Attach ``tier`` / ``screen_notes`` / ``score_companion`` / ``score_series`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Candidate DataFrame as built by ``_candidates_to_dataframe`` in
        ``app.py``. Must contain the columns ``mz``, ``intensity``,
        ``n_galnac``, ``n_gal``, ``ion``, ``mz_diff``. An empty or
        non-matching DataFrame is returned unchanged (with the four
        new columns added as empty strings / 0).
    spectrum_peaks : Sequence[Peak]
        The full peak list of the spectrum the candidates came from.
        Used for neighbour-peak lookup, the series detector, and the
        windowed noise floor.
    da_tol : float
        Symmetric Dalton tolerance for the neighbour/series/envelope
        lookups AND the PURGE gate. Default 0.7 matches the Streamlit
        sidebar default. Da is the binding tolerance; no ppm fallback
        exists. The PURGE gate fires when ``|mz_diff| > da_tol`` --
        0.05 Da at m/z 4000 corresponds to 12.5 ppm, but that is not
        surfaced to the user.
    strictness : str
        Reserved for future preset tuning. Currently ignored (always
        behaves as ``"strict"`` -- GREEN requires companion + series +
        envelope; envelope skip is treated as "envelope not checked"
        rather than permissive).
    spectrum : Spectrum | None
        Optional Spectrum object whose ``upper_bound()`` (acquisition
        upper bound) is used for off-edge detection. When ``None``
        (the default, for back-compat with older callers), the upper
        bound falls back to ``max(peak.mz)`` of ``spectrum_peaks``.
        Pass the actual Spectrum -- with ``mz_hi`` set from the
        user's m/z slider -- so a sparse spectrum whose highest
        peak happens to be far from the acquisition window's upper
        limit doesn't over-penalise high-m/z candidates whose
        companion offsets would land in the un-acquired tail.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the same data plus four new columns.
        The original DataFrame is not mutated.
    """
    out = df.copy()
    required = {"mz", "intensity", "n_galnac", "n_gal", "ion", "mz_diff"}
    if out.empty or not required.issubset(out.columns) or strictness == "off":
        # Off / empty path: still attach the four columns so downstream
        # code does not have to special-case the off state. The score
        # columns default to 0 (not NaN) so sort/export code can
        # treat them as numeric. ``tier`` and ``screen_notes`` default
        # to "" -- they have no meaningful value when the screener
        # hasn't run.
        if "tier" not in out.columns:
            out["tier"] = ""
        if "screen_notes" not in out.columns:
            out["screen_notes"] = ""
        if "score_companion" not in out.columns:
            out["score_companion"] = 0
        if "score_series" not in out.columns:
            out["score_series"] = 0
        return out

    sorted_mz, sorted_intensity = _sorted_mz_index(spectrum_peaks)

    # The upper edge of the spectrum. Companion / series checks that
    # would land past this point are not counted as failures -- the
    # instrument simply didn't acquire that m/z. This is the primary
    # fix for the high-m/z discrimination problem: a 5800-Da peak in
    # a 5998-Da file cannot have its +203 companion by construction,
    # so we should not penalise it for the missing companion.
    #
    # Prefer the Spectrum's upper_bound() (which honours the user's
    # m/z slider via mz_hi when set) over ``max(sorted_mz)`` -- the
    # latter is a peak-derived upper bound that can drift from the
    # acquisition range when the spectrum is sparse. Falling back to
    # ``max(sorted_mz)`` keeps older callers (no Spectrum supplied)
    # working at the previous level of correctness.
    if spectrum is not None:
        _spec_ub = spectrum.upper_bound()
        mz_max: float = _spec_ub if _spec_ub > 0 else float("inf")
    else:
        mz_max = max(sorted_mz) if sorted_mz else float("inf")

    # Pre-compute the per-bin noise floor once for the whole spectrum
    # so every candidate and every series step gets an O(1) lookup
    # instead of an O(N) bin scan. This is the difference between
    # sub-second and multi-second renders on large spectra.
    noise_cache, noise_fallback = _build_noise_floor_cache(spectrum_peaks)

    tiers: list[str] = []
    notes_col: list[str] = []
    score_companion_col: list[int] = []
    score_series_col: list[int] = []

    for _, row in out.iterrows():
        mz = float(row["mz"])
        n_galnac = int(row["n_galnac"])
        n_gal = int(row["n_gal"])
        ion = str(row["ion"])

        try:
            adduct = _adduct_from_ion(ion)
        except ValueError:
            # Defensive: an unknown ion label (shouldn't happen, but
            # the app pipeline could in principle be expanded). Mark
            # the row as PURGE so it is at least visible in the
            # output and the app does not crash.
            tiers.append("PURGE")
            notes_col.append(f"unknown ion label {ion!r}")
            score_companion_col.append(0)
            score_series_col.append(0)
            continue
        # The "noise floor" used for companion / series is the bin
        # around the candidate m/z. The envelope uses the same floor
        # so all three checks are consistent.
        local_noise = _noise_floor_cached(noise_cache, mz, fallback=noise_fallback)

        # Companion count. The third return value indicates whether
        # at least one offset was skipped because it would have
        # landed past the spectrum end -- this is the high-m/z
        # bias fix that lets the app distinguish "no companion" from
        # "no companion in the acquired m/z range". The
        # ``noise_cache`` is passed so each companion offset is
        # checked against the noise floor at ITS m/z, not at the
        # candidate's m/z (which can be very different for offsets
        # of 162-365 Da, especially in spectra with non-uniform
        # background). ``noise_fallback`` is the global min positive
        # intensity so a missing-bin lookup in the per-offset path
        # returns the same value the per-call ``_noise_floor_for_mz``
        # would compute.
        n_companions, comp_notes, comp_skipped = _score_companion(
            mz, sorted_mz, sorted_intensity, da_tol, local_noise,
            mz_max=mz_max, noise_cache=noise_cache,
            companion_tol_da=DEFAULT_COMPANION_TOL_DA,
            noise_fallback=noise_fallback,
        )
        # Series length. Third return value: True if at least one
        # ladder was truncated by the mz_max check.
        series_len, series_notes, series_skipped = _score_series(
            mz, sorted_mz, sorted_intensity, da_tol, noise_cache,
            mz_max=mz_max, companion_tol_da=DEFAULT_COMPANION_TOL_DA,
            noise_fallback=noise_fallback,
        )
        # Envelope. Pass the user's Da tolerance through so it binds
        # at all m/z values -- the companion and series checks already
        # use max(da_tol, companion_tol_da); the envelope check uses
        # da_tol directly. The tolerance is in Daltons: the user's
        # sidebar slider sets it uniformly across the whole m/z range.
        envelope_ok, env_note, envelope_skipped = _score_envelope(
            n_galnac, n_gal, adduct, sorted_mz, sorted_intensity, da_tol, mz,
            noise_at_target=local_noise,
        )

        # Da-only PURGE: |mz_diff| > da_tol -> drop. The user's Da slider is the
        # binding tolerance; there is no ppm fallback. 0.05 Da at m/z 4000 corresponds
        # to 12.5 ppm, but we don't surface that to the user.
        try:
            _mz_diff = abs(float(row.get("mz_diff", 0.0)))
        except (TypeError, ValueError):
            _mz_diff = 0.0
        tolerance_ok = _mz_diff <= da_tol
        _gate_label = f"|m/z diff|={_mz_diff:.3f} Da"
        _gate_limit = f"{da_tol:g} Da"

        # M+0 S/N gate. The strongest peak in the candidate's m/z
        # window is the M+0. If it is below ``MIN_M0_SNR_FOR_KEEP``
        # times the local noise floor the peak itself is not
        # distinguishable from background, so the candidate is
        # PURGEd regardless of any companion/series/envelope
        # evidence -- matches at offsets could be coincidental.
        m0_intensity, m0_mz_observed = _strongest_peak_near(
            sorted_mz, sorted_intensity, mz, da_tol,
        )
        _snr = m0_intensity / local_noise if local_noise > 0 else float("inf")
        m0_sn_ok = _snr >= MIN_M0_SNR_FOR_KEEP

        tier = _tier_from_scores(
            has_companion=n_companions >= 1,
            series_len=series_len,
            envelope_ok=envelope_ok,
            tolerance_ok=tolerance_ok,
            companion_skipped=comp_skipped,
            series_skipped=series_skipped,
            envelope_skipped=envelope_skipped,
            m0_sn_ok=m0_sn_ok,
        )

        # Compose the human-readable note. Report the result of
        # EVERY check so the user can see why a candidate ended
        # up at its tier. For companion and series we report one
        # of three states: found, skipped (off-spectrum), or
        # checked-but-not-found ("none in range"). Without the
        # "none in range" branch, a high-m/z candidate whose
        # companions/series were all checked (in range) but not
        # found would show only the envelope result -- making
        # it look like the companion/series checks weren't run.
        parts: list[str] = []
        if not m0_sn_ok:
            parts.append(
                f"M+0 below {MIN_M0_SNR_FOR_KEEP:g}x noise floor "
                f"(intensity {m0_intensity:.0f}, floor {local_noise:.0f}, "
                f"S/N={_snr:.2f}) (PURGE)"
            )
        if not tolerance_ok:
            parts.append(f"{_gate_label} > {_gate_limit} (PURGE)")
        # Companion / series notes use a YES / NO / skipped triplet
        # so the user can scan the table and see at a glance which
        # checks passed and which returned empty. The old lowercase
        # "companions: +1 Gal" / "companions: none in range" pair
        # made every row look similar even when the underlying
        # answer was different.
        if n_companions:
            parts.append(
                f"Companions: YES ({n_companions} found: "
                + ", ".join(comp_notes) + ")"
            )
        elif comp_skipped:
            parts.append("Companions: skipped (off-spectrum)")
        elif comp_notes:
            # Near-miss case: no companion landed in the search
            # window, but _score_companion found a peak just
            # outside (within 2x the window). Surface those
            # near-miss details so the user can see "peak at X is
            # 1.2 Da off the theoretical +Gal position" and decide
            # whether the offset is real-but-shifted.
            parts.append(
                "Companions: NO (near miss: "
                + "; ".join(comp_notes)
                + ")"
            )
        else:
            parts.append("Companions: NO (none in range)")
        if series_len >= SERIES_MIN_LENGTH:
            parts.append(
                f"Series: YES ({series_len} steps: "
                + ", ".join(series_notes) + ")"
            )
        elif series_skipped:
            parts.append("Series: skipped (off-spectrum)")
        else:
            parts.append("Series: NO (none in range)")
        if env_note:
            parts.append(env_note)

        tiers.append(tier)
        notes_col.append("; ".join(parts))
        score_companion_col.append(n_companions)
        score_series_col.append(series_len)

    out["tier"] = tiers
    out["screen_notes"] = notes_col
    out["score_companion"] = score_companion_col
    out["score_series"] = score_series_col
    # GalNAc-overlap erase rule: walk the survivors and PURGE any
    # second candidate whose M+0/M+1 is shadowed by a GalNAc-dominant
    # neighbour's M+1/M+2/M+3. Runs BEFORE the within-cluster dedup
    # so an "erased" row stays erased (PURGE) and is not just
    # shadowed by the lower m/z (which would hide the erase reason
    # from the user). See ``_erase_overlapped_neighbors`` for the
    # rationale and bounds.
    out = _erase_overlapped_neighbors(out, da_tol=da_tol)
    out = _dedup_nearby_observed_mz(out)
    return out


def _dedup_nearby_observed_mz(
    df: pd.DataFrame,
    *,
    window_da: float = CANDIDATE_MZ_DEDUP_DA,
) -> pd.DataFrame:
    """Drop higher-m/z candidates that sit within ``window_da`` of a
    lower-m/z non-PURGE candidate.

    The user's rule: when two candidates BOTH pass the four checks
    (companion, series, envelope, tolerance) and their observed
    m/z's are within ``window_da`` (default 2.2 Da), keep the lower
    m/z. Rationale: a real biological peak at (e.g.) 2000.0 Da and
    a coincidental match at 2001.5 Da are most plausibly the same
    peak drifting in the same noise bin; the lower-m/z observation
    is the cleaner read.

    Implementation: sort by m/z ascending, walk the rows, drop any
    non-PURGE row whose m/z is within ``window_da`` of any earlier
    non-PURGE row already accepted. PURGE rows are skipped in the
    window check (they've already been flagged) but otherwise pass
    through unchanged -- a PURGE row is still useful information
    that the user might want to see in the table even if they
    filter it from the plot.
    """
    if df.empty or "mz" not in df.columns or "tier" not in df.columns:
        return df
    ordered = df.sort_values("mz", kind="mergesort").reset_index(drop=True)
    new_rows: list[int] = []
    last_kept_mz: float | None = None
    for i, row in ordered.iterrows():
        mz = float(row["mz"])
        tier = str(row.get("tier", ""))
        if tier == "PURGE":
            # A PURGE row doesn't shadow a real peak -- the
            # candidate already failed one of the four checks.
            # Pass it through.
            new_rows.append(i)
            continue
        if last_kept_mz is not None and abs(mz - last_kept_mz) <= window_da:
            # Shadowed by the earlier non-PURGE candidate. Drop
            # the row -- the user asked "pick the lowest m/z", not
            # "annotate the higher one".
            continue
        new_rows.append(i)
        last_kept_mz = mz
    return ordered.iloc[new_rows].reset_index(drop=True)


#: Substring the new "GalNAc-overlap" erase rule uses to mark an
#: erased row in :data:`screen_notes`. The app's "Hide if 13C envelope
#: not observed" filter is independent of this mark; the user can
#: still see the erased row in the table and read the reason.
GALNAC_OVERLAP_ERASE_NOTE: Final[str] = "erased: shadowed by GalNAc-dominant candidate's M+ isotopes"


def _erase_overlapped_neighbors(
    df: pd.DataFrame,
    *,
    da_tol: float,
    window_da: float = GALNAC_OVERLAP_ERASE_DA,
) -> pd.DataFrame:
    """PURGE second candidates whose M+0/M+1 overlaps a GalNAc-dominant neighbour's M+ isotopes.

    The user's rule: when a GalNAc-dominant candidate (``n_galnac >
    n_gal``) sits within ``window_da`` of a higher-m/z candidate AND
    any of the first's M+1, M+2, or M+3 isotope positions falls
    within ``da_tol`` of the second's M+0 OR M+1 position, the second
    candidate is erased. The first's M+k peaks are real signal (the
    GalNAc-dominant envelope rule already requires M+1+M+2+M+3 above
    noise for non-PURGE rows), and a second candidate sitting inside
    the first's isotope envelope cannot be distinguished from those
    satellites -- its "M+0" may be the first's "M+1", its "M+1" may
    be the first's "M+2".

    Implementation: walk the rows in m/z order. For each non-PURGE
    candidate with ``n_galnac > n_gal``, mark every later non-PURGE
    candidate within ``window_da`` whose M+0 or M+1 position is
    within ``da_tol`` of any of the first's M+1/M+2/M+3 positions.
    The marked candidates are PURGEd and a note explaining the
    erasure is appended to ``screen_notes`` so the user can see the
    reason in the table. Already-PURGE rows pass through unchanged
    (they're already filtered).

    This runs BEFORE :func:`_dedup_nearby_observed_mz` so the
    "erase" outcome is the PURGE state (visible in the table with
    its reason in ``screen_notes``) and is not collapsed into
    "shadowed by lower m/z" by the subsequent dedup pass. The dedup
    pass still applies to candidates that were NOT erased, so two
    close non-PURGE candidates where the first is not
    GalNAc-dominant still get collapsed to the lower m/z.
    """
    if df.empty or "mz" not in df.columns or "n_galnac" not in df.columns or "tier" not in df.columns:
        return df
    out = df.copy()
    ordered_idx = out.sort_values("mz", kind="mergesort").index.tolist()
    erasures: dict[int, str] = {}
    for i_pos, i in enumerate(ordered_idx):
        i_tier = str(out.at[i, "tier"])
        if i_tier == "PURGE":
            continue
        i_n_galnac = int(out.at[i, "n_galnac"])
        i_n_gal = int(out.at[i, "n_gal"])
        if i_n_galnac <= i_n_gal:
            continue
        i_mz = float(out.at[i, "mz"])
        # First candidate is GalNAc-dominant and non-PURGE -- its M+1,
        # M+2, M+3 must already be above noise (envelope rule). Walk
        # later candidates within window_da and erase any whose M+0
        # or M+1 is within da_tol of one of those isotope positions.
        m1_pos = i_mz + _C13_MASS_SHIFT
        m2_pos = i_mz + 2.0 * _C13_MASS_SHIFT
        m3_pos = i_mz + 3.0 * _C13_MASS_SHIFT
        first_positions = (m1_pos, m2_pos, m3_pos)
        for j in ordered_idx[i_pos + 1:]:
            j_tier = str(out.at[j, "tier"])
            if j_tier == "PURGE":
                continue
            j_mz = float(out.at[j, "mz"])
            if j_mz - i_mz > window_da:
                break
            # Second candidate's M+0 is at j_mz; M+1 is at j_mz + 1.003.
            # Check whether any first-M+k position falls within da_tol
            # of EITHER second position. If yes, the second is erased.
            j_m1_pos = j_mz + _C13_MASS_SHIFT
            for k_pos in first_positions:
                if abs(k_pos - j_mz) <= da_tol or abs(k_pos - j_m1_pos) <= da_tol:
                    erasures[j] = (
                        f"{GALNAC_OVERLAP_ERASE_NOTE} "
                        f"({i_mz:.2f} Da, n_galnac={i_n_galnac}, n_gal={i_n_gal})"
                    )
                    break
    if not erasures:
        return out
    for idx, note in erasures.items():
        out.at[idx, "tier"] = "PURGE"
        existing = str(out.at[idx, "screen_notes"]) if "screen_notes" in out.columns else ""
        out.at[idx, "screen_notes"] = (existing + "; " + note) if existing else note
    return out


# --- Filter predicates -----------------------------------------------------


#: Substrings that confirm the 13C envelope is acceptable. The
#: app.py "Hide if 13C envelope not observed" filter keeps rows
#: whose ``screen_notes`` contain one of these substrings. The
#: active-failure messages ("13C envelope requires M+1", "13C
#: envelope partial", "13C envelope not observed") must NOT be in
#: this set -- those are the rows the user wants dropped.
_ENVELOPE_OK_NOTE_SUBSTRINGS: tuple[str, ...] = (
    "13C envelope observed",
    "13C M+0 present, no satellite above threshold",
)


def envelope_note_is_ok(notes: str) -> bool:
    """True iff ``notes`` indicates the 13C envelope is acceptable.

    Used by the Streamlit "Hide if 13C envelope not observed" filter
    in app.py. The previous implementation used a loose "M+0 present"
    substring match, which incorrectly kept rows whose note said
    "13C envelope requires M+1 (only M+0 present at 1681.4 m/z)" --
    that note contains the literal phrase "M+0 present" but the
    envelope actively failed (no M+1 found above noise).
    """
    if not notes:
        return False
    return any(s in notes for s in _ENVELOPE_OK_NOTE_SUBSTRINGS)


#: Substring that marks a row as having failed the M+0 S/N gate.
#: The "Hide if M+0 below 1.5x noise" checkbox in app.py drops rows
#: whose ``screen_notes`` contain this substring. The gate fires when
#: the candidate's M+0 intensity is below :data:`MIN_M0_SNR_FOR_KEEP`
#: times the local noise floor.
LOW_SN_PURGE_NOTE_SUBSTRING: Final[str] = "M+0 below "


def is_low_sn_purge(notes: str) -> bool:
    """True iff ``notes`` indicates the candidate was PURGEd for low S/N.

    Used by the Streamlit "Hide if M+0 below 1.5x noise" filter. The
    screener reports the low-S/N PURGE with the substring
    "M+0 below 1.5x noise floor" (the value of
    :data:`MIN_M0_SNR_FOR_KEEP`); this predicate is a thin
    substring check so the filter contract is one line and
    testable.
    """
    if not notes:
        return False
    return LOW_SN_PURGE_NOTE_SUBSTRING in notes
