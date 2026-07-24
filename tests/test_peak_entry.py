from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app import (
    _closest_composition_rows,
    _is_spectrum_analysis_key,
    _label_from_scoped_state_key,
    _merge_manual_peak_candidates,
    _peak_entries_to_peaks,
    _peak_text_to_peaks,
    _spectrum_with_added_peaks,
)
from glycan_ms.core import Adduct, Peak, Spectrum


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


def test_colon_peak_text_accepts_every_numeric_row() -> None:
    peaks, errors = _peak_text_to_peaks(
        "1298:10000\n2354:9000\n2850:8000\n4774:7000\n4205:6000"
    )

    assert errors == []
    assert len(peaks) == 5
    assert [peak.mz for peak in peaks] == [1298, 2354, 2850, 4774, 4205]
    assert [peak.intensity for peak in peaks] == [10000, 9000, 8000, 7000, 6000]


def test_plain_peak_text_uses_default_intensity_and_keeps_valid_rows() -> None:
    peaks, errors = _peak_text_to_peaks("1298\nbad\n1460.367")

    assert peaks == [
        Peak(mz=1298.0, intensity=1.0),
        Peak(mz=1460.367, intensity=1.0),
    ]
    assert len(errors) == 1


def test_closest_composition_returns_one_yes_row_per_peak() -> None:
    peaks, _ = _peak_text_to_peaks(
        "1298:10000\n2354:9000\n2850:8000\n4774:7000\n4205:6000"
    )

    result = _closest_composition_rows(peaks, [Adduct.NA, Adduct.K])

    assert len(result) == 5
    assert result["accepted"].tolist() == ["YES"] * 5
    assert result["mz"].tolist() == [1298, 2354, 2850, 4774, 4205]
    assert result["intensity"].tolist() == [10000, 9000, 8000, 7000, 6000]
    assert result["theoretical_mz"].notna().all()


def test_manual_peak_merge_never_drops_rows_outside_tolerance() -> None:
    peaks, _ = _peak_text_to_peaks(
        "1298:10000\n2354:9000\n2850:8000\n4774:7000\n4205:6000"
    )

    merged = _merge_manual_peak_candidates(
        pd.DataFrame(),
        peaks,
        [Adduct.NA, Adduct.K],
        peaks,
        da_tol=0.01,
        strictness="strict",
    )

    assert len(merged) == 5
    assert merged["accepted"].tolist() == ["YES"] * 5
    assert merged["mz"].tolist() == [1298, 2354, 2850, 4774, 4205]

def test_manual_peak_ui_keeps_all_colon_rows_and_intensities() -> None:
    app_file = Path(__file__).resolve().parents[1] / "app.py"
    values = "1298:10000\n2354:9000\n2850:8000\n4774:7000\n4205:6000"
    app_test = AppTest.from_file(str(app_file)).run(timeout=30)

    app_test.text_area(key="adhoc_check_mz_widget").input(values)
    app_test.button(key="analyse_manual_list").click().run(timeout=30)

    assert not app_test.exception
    result = app_test.dataframe[0].value
    assert len(result) == 5
    assert result["accepted"].tolist() == ["YES"] * 5
    assert result["intensity"].tolist() == [10000, 9000, 8000, 7000, 6000]

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
    assert _label_from_scoped_state_key(
        f"add_peaks_bulk::{label}::2",
        "add_peaks_bulk::",
    ) == label
    assert _label_from_scoped_state_key(
        f"manual_peak_entries::{label}",
        "manual_peak_entries::",
    ) == label


def test_candidate_removal_keys_extract_multi_sheet_label() -> None:
    label = "sample :: sheet 1"

    for scope in ("dataset", "active", "compare"):
        assert _label_from_scoped_state_key(
            f"removed_candidates::{scope}::{label}::abc123",
            "removed_candidates::",
        ) == label
