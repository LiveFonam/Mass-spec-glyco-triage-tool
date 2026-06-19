"""Core types and the composition-solver algorithm.

Composition formula (monoisotopic, neutral residue masses):
    M = n*GalNAc + m*Gal + H2O + adduct
where adduct is one of H+ (1.0073), Na+ (22.9898), or K+ (38.9637).

In positive-ion mode the mass spectrometer measures the protonated or
adducted species, so for an observed m/z we solve for the neutral mass
    M_neutral = observed_mz - adduct
and require
    M_neutral = n*203.0794 + m*162.0528 + 18.0106  (within a Da tolerance).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# Monoisotopic residue masses (ExPASy / GlycoMod standard)
GALNAC: float = 203.0794   # HexNAc residue (GalNAc / GlcNAc)
GAL: float = 162.0528      # Hex residue (Gal / Glc / Man)
H2O: float = 18.0106       # free reducing-end water


class Adduct(str, Enum):
    H = "H+"
    NA = "Na+"
    K = "K+"

    @property
    def mass(self) -> float:
        return _ADDUCT_MASS[self]


_ADDUCT_MASS: dict[Adduct, float] = {
    Adduct.H: 1.0073,
    Adduct.NA: 22.9898,
    Adduct.K: 38.9637,
}


@dataclass(frozen=True)
class Peak:
    """A single (m/z, intensity) measurement."""
    mz: float
    intensity: float
    scan_id: int | None = None  # populated when read from mzXML


@dataclass(frozen=True)
class Spectrum:
    """A list of peaks, optionally tagged with a source filename.

    ``mz_hi`` is the spectrum's acquisition upper bound -- the highest
    m/z the instrument actually scanned. When set, callers that need an
    "upper edge" of the spectrum (e.g. the screener's off-edge check)
    should prefer ``mz_hi`` over ``max(peak.mz)``: a sparse spectrum
    whose highest peak happens to be far from the acquisition window's
    upper limit would over-penalise high-m/z candidates whose companion
    offsets would land just past the highest peak (but still inside
    the acquisition range).

    When ``mz_hi`` is ``None`` the upper bound is derived from the
    peak list via :meth:`upper_bound`.
    """
    peaks: list[Peak]
    source: str = ""
    mz_hi: float | None = None

    def __iter__(self):  # type: ignore[override]
        return iter(self.peaks)

    def __len__(self) -> int:
        return len(self.peaks)

    def upper_bound(self) -> float:
        """Return the spectrum's upper m/z bound.

        Prefers the explicitly-set ``mz_hi`` (the user's acquisition
        window); falls back to ``max(peak.mz)`` when unset; returns
        ``0.0`` for an empty peak list so the caller never has to
        special-case the empty spectrum.
        """
        if self.mz_hi is not None:
            return self.mz_hi
        if not self.peaks:
            return 0.0
        return max(p.mz for p in self.peaks)


@dataclass(frozen=True)
class Candidate:
    """A proposed composition for a measured peak."""
    n_galnac: int          # number of GalNAc residues
    n_gal: int             # number of Gal residues
    adduct: Adduct
    neutral_mass: float    # n*203.0794 + m*162.0528 + 18.0106
    theoretical_mz: float  # neutral_mass + adduct.mass
    observed_mz: float     # measured
    ppm_error: float       # signed ppm error vs. observed mz
    mz_diff: float         # observed mz - theoretical mz in Da (signed)
    intensity: float


def ppm_error(observed: float, theoretical: float) -> float:
    """Signed ppm error: (observed - theoretical) / theoretical * 1e6.

    Returns a positive value when the measurement is heavier than the
    proposed composition, negative when lighter.
    """
    if theoretical <= 0:
        raise ValueError("theoretical mass must be positive")
    return (observed - theoretical) / theoretical * 1_000_000.0


def _bounds_for_mz(mz_lo: float, mz_hi: float, adduct: Adduct) -> tuple[int, int, int, int]:
    """Compute (n_min, n_max, m_min, m_max) for residues in a mass window.

    The neutral mass is mz - adduct. We bracket the count of each residue
    from below using the heavier residue (GalNAc) and from above using the
    lighter one (Gal) to make sure the window cannot be undersized.
    """
    neutral_lo = max(0.0, mz_lo - adduct.mass)
    neutral_hi = max(0.0, mz_hi - adduct.mass)
    # minus the reducing-end water
    neutral_lo = max(0.0, neutral_lo - H2O)
    neutral_hi = max(0.0, neutral_hi - H2O)
    n_min = 0
    n_max = int(neutral_hi // GALNAC) + 1
    m_min = 0
    m_max = int(neutral_hi // GAL) + 1
    return n_min, n_max, m_min, m_max


def solve_peak(
    mz: float,
    intensity: float,
    *,
    da_tol: float = 0.7,
    mz_lo: float = 600.0,
    mz_hi: float = 5000.0,
    adducts: list[Adduct] | None = None,
) -> list[Candidate]:
    """Return all glycan compositions matching the observed m/z within tolerance.

    Parameters
    ----------
    mz : float
        Observed m/z value.
    intensity : float
        Carried onto each Candidate for output.
    da_tol : float
        Symmetric tolerance in Daltons.
    mz_lo, mz_hi : float
        Restrict the search to peaks in this m/z range.
    adducts : list[Adduct] | None
        Which adducts to consider. Defaults to all three (H, Na, K).
    """
    if adducts is None:
        adducts = list(Adduct)

    if mz < mz_lo or mz > mz_hi:
        return []

    candidates: list[Candidate] = []
    for adduct in adducts:
        n_min, n_max, m_min, m_max = _bounds_for_mz(mz_lo, mz_hi, adduct)
        for n in range(n_min, n_max + 1):
            for m in range(m_min, m_max + 1):
                neutral = n * GALNAC + m * GAL + H2O
                theo_mz = neutral + adduct.mass
                err = ppm_error(mz, theo_mz)
                if abs(mz - theo_mz) <= da_tol:
                    candidates.append(
                        Candidate(
                            n_galnac=n,
                            n_gal=m,
                            adduct=adduct,
                            neutral_mass=neutral,
                            theoretical_mz=theo_mz,
                            observed_mz=mz,
                            ppm_error=err,
                            mz_diff=mz - theo_mz,
                            intensity=intensity,
                        )
                    )
    # Sort by absolute m/z difference so the best match comes first.
    candidates.sort(key=lambda c: (abs(c.mz_diff), c.n_galnac + c.n_gal))
    return candidates


def solve_spectrum(
    spectrum: Spectrum,
    *,
    da_tol: float = 0.7,
    mz_lo: float = 600.0,
    mz_hi: float = 5000.0,
    adducts: list[Adduct] | None = None,
    min_intensity: float = 0.0,
) -> list[Candidate]:
    """Run the composition solver over every peak in a Spectrum.

    Peaks below ``min_intensity`` (or with negative intensity) are
    skipped to keep output small. The implementation is a thin
    wrapper around the NumPy-vectorized solver in
    :mod:`glycan_ms.solver_fast`, which is the same algorithm
    expressed with ``np.add.outer`` + a broadcast mask. Empirically
    ~70x faster than the original nested-Python version on a 200k
    peak spectrum, with bit-identical candidate output.
    """
    # Deferred import to avoid a circular dependency at module load:
    # ``solver_fast`` imports types from this module, so a top-level
    # import would re-enter ``core`` during its initialisation.
    from .solver_fast import solve_spectrum_vectorized

    if adducts is None:
        adducts = list(Adduct)
    return solve_spectrum_vectorized(
        spectrum,
        da_tol=da_tol,
        mz_lo=mz_lo,
        mz_hi=mz_hi,
        adducts=adducts,
        min_intensity=min_intensity,
    )


def theoretical_mz(n_galnac: int, n_gal: int, adduct: Adduct) -> float:
    """Return the theoretical m/z for a given composition.

    Equivalent to building the neutral mass from the residue formulas and
    adding the adduct mass:
        m/z = n*GALNAC + m*GAL + H2O + adduct.mass
    """
    return n_galnac * GALNAC + n_gal * GAL + H2O + adduct.mass


def nearest_compositions(
    target_mz: float,
    adducts: list[Adduct],
    *,
    n_lo: int = 0,
    n_hi: int = 20,
    m_lo: int = 0,
    m_hi: int = 20,
    exclude: tuple[int, int, Adduct] | None = None,
) -> list[tuple[int, int, Adduct, float]]:
    """Return the compositions closest to ``target_mz`` in m/z space.

    Scans the (n_galnac, n_gal) grid in [n_lo, n_hi] x [m_lo, m_hi] for
    every requested adduct and returns ALL of them, sorted by absolute
    m/z difference from the target. The caller picks how many to show
    (the UI uses 2 alternatives plus the user's chosen composition).

    Parameters
    ----------
    target_mz : float
        Observed m/z to compare against.
    adducts : list[Adduct]
        Which adducts to consider.
    n_lo, n_hi, m_lo, m_hi : int
        Bounding box for the residue counts.
    exclude : tuple | None
        Optional (n, m, adduct) triple to drop from the results
        (typically the user's chosen composition so the "alternatives"
        list does not include itself).

    Returns
    -------
    list[tuple[int, int, Adduct, float]]
        Sorted ascending by |target - theoretical_mz|. Each entry is
        (n_galnac, n_gal, adduct, theoretical_mz).
    """
    hits: list[tuple[int, int, Adduct, float]] = []
    for adduct in adducts:
        for n in range(n_lo, n_hi + 1):
            for m in range(m_lo, m_hi + 1):
                if exclude is not None and (n, m, adduct) == exclude:
                    continue
                th = theoretical_mz(n, m, adduct)
                hits.append((n, m, adduct, th))
    hits.sort(key=lambda h: abs(target_mz - h[3]))
    return hits


def dedup_peaks(
    peaks: list[Peak],
    *,
    bin_width: float = 0.01,
) -> list[Peak]:
    """Collapse peaks that fall in the same m/z bin, keeping the highest intensity.

    Many instrument exports (mzXML, mzML, Bruker CSV) repeat the same
    nominal mass across multiple scans or channels. Each duplicate
    spawns its own set of composition candidates, which inflates the
    output and obscures real hits. This helper rounds each ``mz`` to
    the nearest ``bin_width`` (default 0.01 Da, i.e. 10 mDa) and keeps
    only the peak with the highest ``intensity`` in each bin.

    The bin width is intentionally much smaller than the user's
    matching tolerance (default 0.7 Da) so real, distinct peaks are
    never merged. ``scan_id``, when present, is taken from the
    surviving peak.

    Parameters
    ----------
    peaks : list[Peak]
        Input peak list, any order, may contain duplicates.
    bin_width : float
        Width of the m/z bin in Daltons. Default 0.01 (10 mDa).

    Returns
    -------
    list[Peak]
        Deduplicated peak list, sorted ascending by m/z.
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if not peaks:
        return []
    bins: dict[int, Peak] = {}
    for p in peaks:
        key = int(round(p.mz / bin_width))
        existing = bins.get(key)
        if existing is None or p.intensity > existing.intensity:
            bins[key] = p
    return sorted(bins.values(), key=lambda p: p.mz)
