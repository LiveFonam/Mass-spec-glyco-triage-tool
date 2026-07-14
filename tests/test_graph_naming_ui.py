from __future__ import annotations

import ast
from pathlib import Path

from app import _label_from_scoped_state_key


APP = Path(__file__).resolve().parent.parent / "app.py"


def _render_spectrum() -> ast.FunctionDef:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_render_spectrum":
            return node
    raise AssertionError("_render_spectrum was not found")


def _source_segment(node: ast.AST) -> str:
    source = APP.read_text(encoding="utf-8")
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_graph_name_widgets_are_scoped_to_render_and_spectrum() -> None:
    function = _render_spectrum()
    expected_keys = {
        "graph_section_name::raw::{key_suffix}::{label}",
        "graph_title::raw::{key_suffix}::{label}",
        "graph_section_name::characteristic::{key_suffix}::{label}",
        "graph_title::characteristic::{key_suffix}::{label}",
        "graph_section_name::proportions::{key_suffix}::{label}",
        "graph_title::proportions::{key_suffix}::{label}",
    }
    actual_keys: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.keyword) or node.arg != "key":
            continue
        text = _source_segment(node.value)
        if "graph_section_name::" in text:
            actual_keys.add(text.removeprefix('f"').removesuffix('"'))
        elif text.startswith("graph_title_keys["):
            graph_kind = text.removeprefix('graph_title_keys["').removesuffix('"]')
            actual_keys.add(f"graph_title::{graph_kind}::{{key_suffix}}::{{label}}")

    assert actual_keys == expected_keys
    active_keys = {key.format(key_suffix="active", label="sample") for key in actual_keys}
    compare_keys = {key.format(key_suffix="compare", label="sample") for key in actual_keys}
    assert active_keys.isdisjoint(compare_keys)


def test_graph_name_settings_and_graph_panels_start_collapsed() -> None:
    function = _render_spectrum()
    expander_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "expander"
    ]
    relevant_calls = [
        call
        for call in expander_calls
        if any(
            name in _source_segment(call)
            for name in (
                "Rename graph sections and titles",
                "raw_section_name",
                "characteristic_section_name",
                "proportion_section_name",
            )
        )
    ]

    assert len(relevant_calls) == 4
    for call in relevant_calls:
        expanded = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "expanded"),
            None,
        )
        assert isinstance(expanded, ast.Constant)
        assert expanded.value is False


def test_custom_graph_titles_are_applied_to_all_three_figures() -> None:
    source = _source_segment(_render_spectrum())

    assert "fig.update_layout(title=raw_graph_title)" in source
    assert "characteristic_fig.update_layout(title=characteristic_graph_title)" in source
    assert "proportion_fig.update_layout(title=proportion_graph_title)" in source


def test_custom_graph_titles_invalidate_prepared_downloads() -> None:
    source = _source_segment(_render_spectrum())
    signature_start = source.index("export_signature = hashlib.sha256")
    signature_end = source.index("_download_buttons(", signature_start)
    signature_source = source[signature_start:signature_end]

    assert "raw_graph_title" in signature_source
    assert "characteristic_graph_title" in signature_source
    assert "proportion_graph_title" in signature_source


def test_default_graph_titles_follow_sample_rename_until_customized() -> None:
    source = _source_segment(_render_spectrum())

    assert "previous_sample_name != new_name" in source
    assert "previous_title_defaults" in source
    assert "st.session_state[title_key] = current_title_defaults[graph_kind]" in source


def test_graph_naming_keys_extract_dataset_label_for_cleanup() -> None:
    label = "file :: sheet"

    assert _label_from_scoped_state_key(
        f"graph_title::raw::active::{label}", "graph_title::"
    ) == label
    assert _label_from_scoped_state_key(
        f"graph_section_name::proportions::compare::{label}",
        "graph_section_name::",
    ) == label
    assert _label_from_scoped_state_key(
        f"graph_title_sample_source::active::{label}",
        "graph_title_sample_source::",
    ) == label
    assert _label_from_scoped_state_key(
        f"galnac_color_v2::active::{label}::3", "galnac_color_v2::"
    ) == label


def test_galnac_picker_uses_versioned_state_key_for_default_migration() -> None:
    source = APP.read_text(encoding="utf-8")

    assert 'picker_key = f"galnac_color_v2::' in source
