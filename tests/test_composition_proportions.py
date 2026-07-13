import pandas as pd
import pytest
import io
import zipfile

from app import (
    _characteristic_peaks_plot,
    _composition_proportion_plot,
    _curated_peak_rows,
    _pngs_to_zip_bytes,
    _with_candidate_row_ids,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"mz": 900.0, "intensity": 10.0, "n_galnac": 1, "n_gal": 4, "total": 5, "ion": "H+", "mz_diff": 0.01},
            {"mz": 922.0, "intensity": 20.0, "n_galnac": 1, "n_gal": 4, "total": 5, "ion": "Na+", "mz_diff": 0.02},
            {"mz": 1100.0, "intensity": 70.0, "n_galnac": 2, "n_gal": 4, "total": 6, "ion": "Na+", "mz_diff": -0.01},
        ]
    )


def test_candidate_row_ids_are_stable_and_unique() -> None:
    first = _with_candidate_row_ids(_candidate_rows())
    second = _with_candidate_row_ids(_candidate_rows())

    assert first["_candidate_row_id"].tolist() == second["_candidate_row_id"].tolist()
    assert first["_candidate_row_id"].is_unique


def test_composition_proportions_combine_ions_and_use_total_signal() -> None:
    figure = _composition_proportion_plot(_candidate_rows(), "Sample")
    traces = {trace.name: trace for trace in figure.data}

    galnac_1 = traces["1 GalNAc + 4 Gal"]
    galnac_2 = traces["2 GalNAc + 4 Gal"]
    assert list(galnac_1.x) == [5, 6]
    assert list(galnac_1.y) == pytest.approx([30.0, 0.0])
    assert list(galnac_2.y) == pytest.approx([0.0, 70.0])
    assert figure.layout.barmode == "stack"
    assert figure.layout.xaxis.dtick == 1


def test_removing_a_row_recalculates_proportions() -> None:
    candidates = _with_candidate_row_ids(_candidate_rows())
    removed_id = candidates.iloc[2]["_candidate_row_id"]
    retained = candidates[candidates["_candidate_row_id"] != removed_id]

    figure = _composition_proportion_plot(retained, "Curated")

    assert len(figure.data) == 1
    assert list(figure.data[0].y) == pytest.approx([100.0])


def test_curated_peaks_keep_strongest_row_within_point_zero_one_mz() -> None:
    candidates = _candidate_rows()
    close_row = candidates.iloc[0].copy()
    close_row["mz"] = 900.01
    close_row["intensity"] = 25.0
    candidates = pd.concat([candidates, close_row.to_frame().T], ignore_index=True)

    curated = _curated_peak_rows(candidates)

    assert len(curated) == 3
    assert float(curated.iloc[0]["mz"]) == pytest.approx(900.01)
    assert float(curated.iloc[0]["intensity"]) == pytest.approx(25.0)


def test_characteristic_peaks_can_normalize_to_one_hundred_percent() -> None:
    figure = _characteristic_peaks_plot(_candidate_rows(), "Sample", normalize=True)
    marker_trace = figure.data[1]

    assert max(marker_trace.y) == pytest.approx(100.0)
    assert figure.layout.dragmode == "zoom"
    assert figure.layout.yaxis.ticksuffix == "%"


@pytest.mark.parametrize("files", [
    {"proportions.png": b"one"},
    {"spectrum.png": b"one", "peaks.png": b"two", "proportions.png": b"three"},
])
def test_png_zip_supports_one_or_multiple_graphs(files: dict[str, bytes]) -> None:
    payload = _pngs_to_zip_bytes("sample graphs", files)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert sorted(archive.namelist()) == sorted(
            f"sample_graphs/{filename}" for filename in files
        )
        for filename, expected in files.items():
            assert archive.read(f"sample_graphs/{filename}") == expected
