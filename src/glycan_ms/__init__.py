"""Glycan mass spectrometry composition analyzer."""

from .core import (
    GALNAC,
    GAL,
    H2O,
    Adduct,
    Peak,
    Spectrum,
    Candidate,
    ppm_error,
    solve_peak,
    solve_spectrum,
    theoretical_mz,
    nearest_compositions,
    dedup_peaks,
)
from .parser_table import parse_table, parse_table_sheets, read_xlsx_sheets
from .parser_mzxml import first_scan_spectrum, parse_mzxml
from .parser_mzml import first_scan_spectrum_mzml, parse_mzml as parse_mzml_stream
from .solver_fast import solve_spectrum_vectorized
from .screener import screen_candidates

__all__ = [
    "GALNAC",
    "GAL",
    "H2O",
    "Adduct",
    "Peak",
    "Spectrum",
    "Candidate",
    "ppm_error",
    "solve_peak",
    "solve_spectrum",
    "solve_spectrum_vectorized",
    "theoretical_mz",
    "nearest_compositions",
    "dedup_peaks",
    "parse_table",
    "parse_table_sheets",
    "read_xlsx_sheets",
    "first_scan_spectrum",
    "parse_mzxml",
    "first_scan_spectrum_mzml",
    "parse_mzml_stream",
    "screen_candidates",
]

__version__ = "0.34.3"
