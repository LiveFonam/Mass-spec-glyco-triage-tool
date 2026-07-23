import pandas as pd
import pytest
import io
import zipfile
from PIL import Image

import app
from app import (
    _candidate_removal_revision,
    _candidate_removal_state_key,
    _characteristic_peaks_plot,
    _composition_proportion_plot,
    _compose_graph_grid_png,
    _curated_peak_rows,
    _graph_cart_zip_bytes,
    _graph_export_filename,
    _galnac_color_map,
    _merge_candidate_removal_ids,
    _pngs_to_zip_bytes,
    _record_candidate_editor_removals,
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


def test_candidate_removal_state_is_shared_by_dataset_not_render_position() -> None:
    label = "sample :: sheet 1"

    shared_key = _candidate_removal_state_key(label, "abc123")

    assert shared_key == "removed_candidates::dataset::sample :: sheet 1::abc123"
    assert _candidate_removal_state_key(label, "other") != shared_key
    assert _candidate_removal_state_key("another sample", "abc123") != shared_key


def test_candidate_removal_revision_tracks_ids_not_only_row_count() -> None:
    assert _candidate_removal_revision(["row-a"]) != _candidate_removal_revision(["row-b"])
    assert _candidate_removal_revision(["row-a"]) == _candidate_removal_revision(["row-a"])


def test_candidate_removal_merge_preserves_order_without_duplicates() -> None:
    assert _merge_candidate_removal_ids(
        ["first", "second"],
        ["second", "third"],
    ) == ["first", "second", "third"]


def test_separate_candidate_editors_commit_both_removals(monkeypatch) -> None:
    session_state = {
        "table_top": {"edited_rows": {1: {"_remove": True}}},
        "table_bottom": {"edited_rows": {"0": {"_remove": True}}},
        "removed_top": [],
        "removed_bottom": ["existing"],
    }
    monkeypatch.setattr(app.st, "session_state", session_state)

    _record_candidate_editor_removals(
        "table_top",
        "removed_top",
        ["top-0", "top-1"],
    )
    _record_candidate_editor_removals(
        "table_bottom",
        "removed_bottom",
        ["bottom-0", "bottom-1"],
    )

    assert session_state["removed_top"] == ["top-1"]
    assert session_state["removed_bottom"] == ["existing", "bottom-0"]


def test_composition_proportions_combine_ions_and_use_total_signal() -> None:
    figure = _composition_proportion_plot(_candidate_rows(), "Sample")
    traces = {trace.name: trace for trace in figure.data}

    galnac_1 = traces["1 GalNAc"]
    galnac_2 = traces["2 GalNAc"]
    assert list(galnac_1.x) == [5, 6]
    assert list(galnac_1.y) == pytest.approx([30.0, 0.0])
    assert list(galnac_2.y) == pytest.approx([0.0, 70.0])
    assert figure.layout.barmode == "stack"
    assert figure.layout.xaxis.dtick == 1


def test_composition_colors_encode_galnac_count_not_exact_composition() -> None:
    candidates = pd.concat(
        [
            _candidate_rows(),
            pd.DataFrame([
                {"mz": 1250.0, "intensity": 15.0, "n_galnac": 1, "n_gal": 5, "total": 6, "ion": "Na+", "mz_diff": 0.01},
            ]),
        ],
        ignore_index=True,
    )

    figure = _composition_proportion_plot(candidates, "Sample")
    one_galnac_traces = [trace for trace in figure.data if trace.name == "1 GalNAc"]

    assert len(one_galnac_traces) == 2
    assert len({trace.marker.color for trace in one_galnac_traces}) == 1
    assert sum(bool(trace.showlegend) for trace in one_galnac_traces) == 1
    assert figure.layout.legend.title.text == "GalNAc count"


def test_proportion_labels_stay_horizontal_and_shrink_for_small_segments() -> None:
    candidates = pd.DataFrame([
        {"mz": 900.0, "intensity": 95.0, "n_galnac": 1, "n_gal": 4, "total": 5, "ion": "H+"},
        {"mz": 922.0, "intensity": 5.0, "n_galnac": 2, "n_gal": 3, "total": 5, "ion": "Na+"},
    ])

    figure = _composition_proportion_plot(candidates, "Sample")

    assert all(trace.textangle == 0 for trace in figure.data)
    sizes = {trace.name: trace.textfont.size[0] for trace in figure.data}
    assert sizes["2 GalNAc"] < sizes["1 GalNAc"]
    assert figure.layout.uniformtext.minsize == 6


def test_default_galnac_colors_are_distinct() -> None:
    colors = _galnac_color_map([0, 1, 2, 3])

    assert colors == {
        0: "#8B0000",
        1: "#F4A3A3",
        2: "#0072B2",
        3: "#009E73",
    }


def test_default_galnac_colors_do_not_shift_when_counts_are_missing() -> None:
    assert _galnac_color_map([1, 3]) == {
        1: "#F4A3A3",
        3: "#009E73",
    }


def test_gradient_galnac_colors_use_increasing_concentration() -> None:
    colors = _galnac_color_map([1, 2, 3], mode="gradient")

    assert colors[3] == "#009E73"
    brightness = {
        count: sum(int(color[index:index + 2], 16) for index in (1, 3, 5))
        for count, color in colors.items()
    }
    assert brightness[1] > brightness[2] > brightness[3]


def test_distinct_galnac_colors_are_user_overridable() -> None:
    colors = _galnac_color_map(
        [1, 2],
        mode="distinct",
        distinct_colors={1: "#112233", 2: "#445566"},
    )

    assert colors == {1: "#112233", 2: "#445566"}


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


def test_graph_export_filenames_use_requested_suffixes() -> None:
    assert _graph_export_filename("Dataset 1", "spectrum") == "Dataset_1_spec.png"
    assert _graph_export_filename("Dataset 1", "peaks") == "Dataset_1_charpeaks.png"
    assert _graph_export_filename("Dataset 1", "proportions") == "Dataset_1_prograph.png"


def test_global_graph_cart_zips_multiple_datasets_together() -> None:
    cart = {}
    for dataset_id, dataset_name in (("one", "Dataset 1"), ("two", "Dataset 2")):
        for kind in ("spectrum", "peaks", "proportions"):
            cart[f"{dataset_id}::{kind}"] = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "kind": kind,
                "filename": _graph_export_filename(dataset_name, kind),
                "payload": f"{dataset_id}-{kind}".encode(),
            }

    payload = _graph_cart_zip_bytes(cart)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert sorted(archive.namelist()) == sorted(
            f"all_prepared_glycan_graphs/{_graph_export_filename(dataset_name, kind)}"
            for dataset_name in ("Dataset 1", "Dataset 2")
            for kind in ("spectrum", "peaks", "proportions")
        )


def _solid_png(color: str, size: tuple[int, int] = (400, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_composite_png_puts_two_graphs_side_by_side_with_a_title() -> None:
    payload = _compose_graph_grid_png(
        [_solid_png("red"), _solid_png("blue")],
        "Shared title",
    )

    with Image.open(io.BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.width > image.height
        assert image.width >= 800


def test_composite_png_uses_two_rows_for_three_or_four_graphs() -> None:
    three = _compose_graph_grid_png([_solid_png("red")] * 3, "Three")
    four = _compose_graph_grid_png([_solid_png("blue")] * 4, "Four")

    with Image.open(io.BytesIO(three)) as three_image, Image.open(io.BytesIO(four)) as four_image:
        assert three_image.height > 400
        assert four_image.size == three_image.size


def test_composite_png_supports_more_than_four_graphs() -> None:
    payload = _compose_graph_grid_png([_solid_png("blue")] * 7, "Seven graphs")

    with Image.open(io.BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.width >= 400
        assert image.height >= 200
