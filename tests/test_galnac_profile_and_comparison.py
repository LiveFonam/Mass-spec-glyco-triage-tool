from __future__ import annotations

import inspect

import pandas as pd

import app
from app import (
    _candidates_to_dataframe,
    _characteristic_peaks_plot,
    _composition_proportion_plot,
    _is_galnac_biased_composition,
    _spectrum_plot,
    _validated_axis_range,
)
from glycan_ms.core import Adduct, Candidate, GAL, GALNAC, H2O, Peak


def _candidate(n_galnac: int, n_gal: int) -> Candidate:
    neutral_mass = n_galnac * GALNAC + n_gal * GAL + H2O
    theoretical_mz = neutral_mass + Adduct.NA.mass
    return Candidate(
        n_galnac=n_galnac,
        n_gal=n_gal,
        adduct=Adduct.NA,
        neutral_mass=neutral_mass,
        theoretical_mz=theoretical_mz,
        observed_mz=theoretical_mz,
        ppm_error=0.0,
        mz_diff=0.0,
        intensity=1000.0,
    )


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mz": 1200.0,
                "intensity": 100.0,
                "n_galnac": 2,
                "n_gal": 1,
                "total": 3,
                "ion": "Na+",
                "mz_diff": 0.0,
            },
            {
                "mz": 1800.0,
                "intensity": 200.0,
                "n_galnac": 3,
                "n_gal": 2,
                "total": 5,
                "ion": "Na+",
                "mz_diff": 0.0,
            },
        ]
    )


def test_galnac_biased_profile_has_explicit_composition_rule() -> None:
    assert _is_galnac_biased_composition(3, 1)
    assert _is_galnac_biased_composition(2, 2)
    assert not _is_galnac_biased_composition(1, 2)
    assert not _is_galnac_biased_composition(0, 0)


def test_galnac_biased_dataframe_filters_standard_candidates() -> None:
    candidates = [
        _candidate(3, 1),
        _candidate(2, 2),
        _candidate(1, 3),
        _candidate(0, 0),
    ]

    standard = _candidates_to_dataframe(candidates)
    biased = _candidates_to_dataframe(candidates, galnac_biased=True)

    assert len(standard) == 4
    assert set(zip(biased["n_galnac"], biased["n_gal"])) == {(3, 1), (2, 2)}


def test_axis_range_validation_requires_finite_increasing_values() -> None:
    assert _validated_axis_range(False, 1000.0, 2000.0) is None
    assert _validated_axis_range(True, 1000.0, 2000.0) == (1000.0, 2000.0)
    assert _validated_axis_range(True, 2000.0, 1000.0) is None
    assert _validated_axis_range(True, float("nan"), 2000.0) is None


def test_fixed_mz_range_is_shared_by_raw_and_characteristic_graphs() -> None:
    expected_range = (1000.0, 2200.0)
    raw = _spectrum_plot(
        [Peak(mz=1200.0, intensity=100.0), Peak(mz=1800.0, intensity=200.0)],
        pd.DataFrame(),
        "Sample",
        mz_min=600.0,
        mz_max=5000.0,
        fixed_x_range=expected_range,
    )
    characteristic = _characteristic_peaks_plot(
        _candidate_rows(),
        "Sample",
        x_range=expected_range,
    )

    assert tuple(raw.layout.xaxis.range) == expected_range
    assert tuple(characteristic.layout.xaxis.range) == expected_range


def test_fixed_dp_range_is_applied_to_composition_graph() -> None:
    figure = _composition_proportion_plot(
        _candidate_rows(),
        "Sample",
        x_range=(0.0, 12.0),
    )

    assert tuple(figure.layout.xaxis.range) == (0.0, 12.0)


def test_compare_ui_accepts_multiple_samples_without_a_selection_cap() -> None:
    source = inspect.getsource(app.main)

    assert "comparison_labels = st.multiselect(" in source
    assert "for comparison_label in comparison_labels:" in source
    assert "max_selections" not in source
    assert "compare_picks_v2::" in source