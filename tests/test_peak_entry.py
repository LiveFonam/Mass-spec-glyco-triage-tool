from __future__ import annotations

import math

import pandas as pd

from app import (
    _is_spectrum_analysis_key,
    _label_from_scoped_state_key,
    _peak_entries_to_peaks,
    _spectrum_with_added_peaks,
)
from glycan_ms.core import Peak, Spectrum


def test_peak_entries_convert_complete_rows_and_ignore_blank_rows() -> None:
    rows = pd.DataFrame(
        {
            "mz": [1177.4236, None, 1460.367],
            "intensity": [1000.0, None, 250.5],
        }
    )

    peaks, errors = _peak_entries_to_peaks(rows)

    assert errors == []
    assert peaks == [
        Peak(mz=1177.4236, intensity=1000.0),
        Peak(mz=1460.367, intensity=250.5),
    ]


def test_peak_entries_report_incomplete_and_invalid_rows() -> None:
    rows = pd.DataFrame(
        {
            "mz": [1177.4, None, -1.0, 1200.0, math.inf],
            "intensity": [None, 10.0, 20.0, -5.0, 30.0],
        }
    )

    peaks, errors = _peak_entries_to_peaks(rows)

    assert peaks == []
    assert errors == [
        "Row 1: enter both m/z and intensity.",
        "Row 2: enter both m/z and intensity.",
        "Row 3: m/z must be a positive finite number.",
        "Row 4: intensity must be a non-negative finite number.",
        "Row 5: m/z must be a positive finite number.",
    ]


def test_spectrum_with_added_peaks_preserves_source_metadata() -> None:
    spectrum = Spectrum(
        peaks=[Peak(mz=1000.0, intensity=50.0)],
        source="sample.csv",
        mz_hi=5000.0,
    )

    updated = _spectrum_with_added_peaks(
        spectrum,
        [Peak(mz=1200.0, intensity=75.0)],
    )

    assert updated.peaks == [
        Peak(mz=1000.0, intensity=50.0),
        Peak(mz=1200.0, intensity=75.0),
    ]
    assert updated.source == "sample.csv"
    assert updated.mz_hi == 5000.0


def test_spectrum_analysis_key_matching_is_scoped_to_label() -> None:
    label = "sample :: sheet 1"

    assert _is_spectrum_analysis_key(
        f"candidates::{label}::abc123",
        label,
    )
    assert _is_spectrum_analysis_key(
        f"spectrum::{label}::abc123",
        label,
    )
    assert _is_spectrum_analysis_key(
        f"removed_candidates::dataset::{label}::abc123",
        label,
    )
    assert _is_spectrum_analysis_key(
        f"removed_candidates::active::{label}::abc123",
        label,
    )
    assert _is_spectrum_analysis_key(
        f"prepared_graph_downloads::compare::{label}",
        label,
    )
    assert not _is_spectrum_analysis_key(
        "candidates::another sample::abc123",
        label,
    )


def test_peak_entry_widget_keys_extract_multi_sheet_label() -> None:
    label = "sample :: sheet 1"

    assert _label_from_scoped_state_key(
        f"add_peaks_editor::{label}::2",
        "add_peaks_editor::",
    ) == label
    assert _label_from_scoped_state_key(
        f"add_peaks_apply::{label}::2",
        "add_peaks_apply::",
    ) == label


def test_candidate_removal_keys_extract_multi_sheet_label() -> None:
    label = "sample :: sheet 1"

    for scope in ("dataset", "active", "compare"):
        assert _label_from_scoped_state_key(
            f"removed_candidates::{scope}::{label}::abc123",
            "removed_candidates::",
        ) == label
