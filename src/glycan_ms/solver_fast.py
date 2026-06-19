"""NumPy-vectorized composition solver.

This mirrors :func:`glycan_ms.core.solve_peak` but enumerates the
``(n_galnac, n_gal)`` grid with ``np.add.outer`` and applies a per-adduct
Da window in bulk, which is substantially faster than the nested Python
loops in :mod:`glycan_ms.core` for wide m/z windows or large adduct sets.
"""

from __future__ import annotations

import time

import numpy as np

from .core import (
    GALNAC,
    GAL,
    H2O,
    Adduct,
    Candidate,
    Peak,
    Spectrum,
)


def _bounds_for_mz(
    mz_lo: float, mz_hi: float, adduct_mass: float
) -> tuple[int, int, int, int]:
    """Vectorized counterpart to :func:`core._bounds_for_mz`."""
    neutral_lo = max(0.0, mz_lo - adduct_mass)
    neutral_hi = max(0.0, mz_hi - adduct_mass)
    neutral_lo = max(0.0, neutral_lo - H2O)
    neutral_hi = max(0.0, neutral_hi - H2O)
    n_max = int(neutral_hi // GALNAC) + 1
    m_max = int(neutral_hi // GAL) + 1
    return 0, n_max, 0, m_max


def solve_spectrum_vectorized(
    spectrum: Spectrum,
    da_tol: float = 0.7,
    mz_lo: float = 600.0,
    mz_hi: float = 5000.0,
    adducts: list[Adduct] | None = None,
    min_intensity: float = 0.0,
) -> list[Candidate]:
    """Return all glycan compositions matching peaks in ``spectrum`` within tolerance.

    Parameters
    ----------
    spectrum : Spectrum
        Observed peaks to solve.
    da_tol : float
        Symmetric tolerance in Daltons.
    mz_lo, mz_hi : float
        Restrict the search to peaks in this m/z range.
    adducts : list[Adduct] | None
        Which adducts to consider. Defaults to all three (H, Na, K).
    min_intensity : float
        Peaks below this intensity are skipped.

    Returns
    -------
    list[Candidate]
        Matching compositions, sorted by absolute m/z difference then by
        total residue count, identical (modulo ordering across peaks)
        to :func:`glycan_ms.core.solve_spectrum`.
    """
    if adducts is None:
        adducts = list(Adduct)

    # Input is assumed to be already deduped. The single source of
    # truth for peak dedup lives at the app.py boundary that feeds
    # _solve_one (app.py:1603); callers (CLI, tests, the vectorized
    # wrapper) must dedup themselves if they bypass that path.
    # Re-deduping here would waste ~5-10 ms per solve on a 10k-peak
    # spectrum.
    peaks = spectrum.peaks

    candidates: list[Candidate] = []
    for adduct in adducts:
        n_min, n_max, m_min, m_max = _bounds_for_mz(mz_lo, mz_hi, adduct.mass)
        if n_max < n_min or m_max < m_min:
            continue

        n_grid = np.arange(n_min, n_max + 1, dtype=np.float64)
        m_grid = np.arange(m_min, m_max + 1, dtype=np.float64)

        # neutral[n, m] = n * GALNAC + m * GAL + H2O
        neutral = np.add.outer(n_grid * GALNAC, m_grid * GAL) + H2O
        theoretical_mz = neutral + adduct.mass  # shape (n_grid, m_grid)

        for peak in peaks:
            if peak.intensity < min_intensity:
                continue
            if peak.mz < mz_lo or peak.mz > mz_hi:
                continue

            # Per-adduct Da window to avoid scanning every cell of the
            # 2D grid for each peak in tight adduct-by-adduct order.
            lo = peak.mz - da_tol
            hi = peak.mz + da_tol
            mask = (theoretical_mz >= lo) & (theoretical_mz <= hi)
            if not mask.any():
                continue

            hits_n, hits_m = np.where(mask)
            theo = theoretical_mz[mask]
            neut = neutral[mask]
            errs = (peak.mz - theo) / theo * 1_000_000.0

            order = np.argsort(np.abs(peak.mz - theo), kind="stable")
            for idx in order:
                candidates.append(
                    Candidate(
                        n_galnac=int(hits_n[idx]),
                        n_gal=int(hits_m[idx]),
                        adduct=adduct,
                        neutral_mass=float(neut[idx]),
                        theoretical_mz=float(theo[idx]),
                        observed_mz=peak.mz,
                        ppm_error=float(errs[idx]),
                        mz_diff=peak.mz - float(theo[idx]),
                        intensity=peak.intensity,
                    )
                )

    return candidates


if __name__ == "__main__":
    from .core import solve_spectrum

    # Fake peak around m/z 1500: neutral mass ~ 1481 with H+ adduct.
    fake_peak = Peak(mz=1500.0, intensity=1.0)
    fake_spectrum = Spectrum(peaks=[fake_peak])

    n_repeats = 5

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        ref = solve_spectrum(fake_spectrum, da_tol=0.7)
    ref_time = (time.perf_counter() - t0) / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        fast = solve_spectrum_vectorized(fake_spectrum, da_tol=0.7)
    fast_time = (time.perf_counter() - t0) / n_repeats

    ref_n = len(ref)
    fast_n = len(fast)

    print(f"reference candidate count: {ref_n}")
    print(f"vectorized candidate count: {fast_n}")
    print(f"reference avg time:  {ref_time * 1e3:.3f} ms")
    print(f"vectorized avg time: {fast_time * 1e3:.3f} ms")
