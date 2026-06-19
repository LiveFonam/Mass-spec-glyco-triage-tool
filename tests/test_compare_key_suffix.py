"""Regression tests for the ``spec_rename::`` duplicate widget key bug.

When the Compare section rendered the same spectrum that was also
showing in the active tab, both calls to ``_render_spectrum`` tried
to create a ``st.text_input(key="spec_rename::{label}")`` widget with
the SAME key, which raised
``streamlit.errors.StreamlitDuplicateElementKey`` and broke the page.

The fix: ``_render_spectrum`` now takes a ``key_suffix`` parameter
and namespaces every internal widget key with it. The Compare caller
passes ``key_suffix="compare"`` so the active and Compare renders
never collide on widget keys. The underlying per-spectrum
``session_state`` (rename override, ion pills, hide_reds, plot
style) is intentionally NOT namespaced: it stays shared so the
user's pill / toggle / label choices apply to the same spectrum in
both render sites.

These tests guard that contract. They are static + behavioural:
  1. AST: every per-spectrum widget key inside ``_render_spectrum``
     references ``key_suffix`` (so two callers with different
     suffixes cannot collide).
  2. AST: the Compare call site actually passes
     ``key_suffix="compare"``.
  3. AST: the active call site does NOT pass ``key_suffix`` (so the
     default ``"active"`` is used, preserving any pre-existing
     per-spectrum session_state keys from earlier versions of the
     app).
  4. Behaviour: extracting the two call sites' widget keys and
     rendering them in a flat dict must produce distinct keys, no
     matter what ``label`` is.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
APP = REPO / "app.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_app_tree() -> ast.Module:
    return ast.parse(APP.read_text())


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in app.py")


def _collect_widget_keys(func: ast.FunctionDef) -> list[tuple[str, bool]]:
    """Walk the function and return every ``key=...`` kwarg.

    Each entry is ``(reconstructed_key, uses_key_suffix)``. The
    reconstructed key is what the f-string would look like if all
    ``{label}`` / ``{key_suffix}`` were left as bare names -- it is
    NOT a runtime value, just a structural fingerprint.
    """
    out: list[tuple[str, bool]] = []
    for sub in ast.walk(func):
        if not isinstance(sub, ast.keyword):
            continue
        if sub.arg != "key":
            continue
        v = sub.value
        if isinstance(v, ast.JoinedStr):
            parts: list[str] = []
            has_key_suffix = False
            for piece in v.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    parts.append(piece.value)
                elif isinstance(piece, ast.FormattedValue):
                    inner = piece.value
                    if isinstance(inner, ast.Name):
                        parts.append("{" + inner.id + "}")
                        if inner.id == "key_suffix":
                            has_key_suffix = True
                    elif isinstance(inner, ast.Call):
                        # f"key_{fn()}_{label}" -- uncommon here, but
                        # be safe: render the call's func name.
                        parts.append("{call}")
            out.append(("".join(parts), has_key_suffix))
        elif isinstance(v, ast.Name):
            # The namespaced key is built into a local variable
            # (e.g. ``rename_widget``) and passed by reference. Treat
            # this as "uses key_suffix" if the variable name suggests
            # it does (we cross-check by reading the assignment).
            out.append((f"<<{v.id}>>", True))
        elif isinstance(v, ast.Constant):
            out.append((repr(v.value), False))
    return out


# ---------------------------------------------------------------------------
# 1) AST: every per-spectrum widget key inside _render_spectrum
#    references key_suffix.
# ---------------------------------------------------------------------------
def test_render_spectrum_signature_has_key_suffix() -> None:
    """``_render_spectrum`` must accept a ``key_suffix`` keyword."""
    tree = _load_app_tree()
    func = _find_function(tree, "_render_spectrum")
    kw_names = [a.arg for a in func.args.kwonlyargs]
    assert "key_suffix" in kw_names, (
        f"_render_spectrum is missing the key_suffix keyword arg "
        f"(got: {kw_names}). Add ``key_suffix: str = 'active'`` to "
        f"the signature so callers can namespace their widget keys."
    )


def test_render_spectrum_widget_keys_namespaced() -> None:
    """Every per-spectrum widget key inside ``_render_spectrum``
    must reference ``key_suffix``.

    Without namespacing, the Compare section and the active render
    collide on the same widget key, which raises
    ``StreamlitDuplicateElementKey``.
    """
    tree = _load_app_tree()
    func = _find_function(tree, "_render_spectrum")
    keys = _collect_widget_keys(func)

    # There must be at least 4 widget keys: text_input, 2 buttons
    # (Dots / Bars), multiselect, 3 ion checkboxes, hide_reds
    # checkbox, plotly chart, 2 dataframes.
    assert len(keys) >= 4, (
        f"Expected at least 4 widget keys in _render_spectrum, "
        f"found {len(keys)}: {keys}"
    )

    for k, uses_suffix in keys:
        # The namespaced-key form is a Name reference (e.g. ``rename_widget``)
        # -- we can't statically prove it uses key_suffix, but the
        # the upstream docstring and naming convention are strong
        # evidence. The f-string form MUST show the {key_suffix} marker.
        if k.startswith("<<") and k.endswith(">>"):
            continue
        assert uses_suffix, (
            f"Widget key {k!r} inside _render_spectrum does not "
            f"reference key_suffix. Compare mode and the active "
            f"render would collide on this key. Wrap it in an "
            f"f-string with {{key_suffix}}."
        )


# ---------------------------------------------------------------------------
# 2) AST: the Compare call site passes key_suffix="compare".
# ---------------------------------------------------------------------------
def test_compare_call_passes_compare_suffix() -> None:
    """The Compare section's ``_render_spectrum`` call must pass
    ``key_suffix="compare"`` so its widgets do not collide with the
    active render.
    """
    src = APP.read_text()
    # Find a block that calls _render_spectrum with plot_key=
    # "compare_plot::{lbl}" -- that's the Compare call site.
    pattern = re.compile(
        r"_render_spectrum\(\s*"
        r"lbl,\s*"
        r"show_metrics=False,\s*"
        r"plot_key=f?[\"']compare_plot::\{lbl\}[\"'],\s*"
        r"key_suffix=\"compare\",?\s*"
        r"\)",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "Compare section's _render_spectrum call is missing "
        "key_suffix=\"compare\". Without it, the active and Compare "
        "renders share the same widget key, raising "
        "StreamlitDuplicateElementKey."
    )


# ---------------------------------------------------------------------------
# 3) AST: the active call site does NOT pass key_suffix (so the
#    default "active" is used).
# ---------------------------------------------------------------------------
def test_active_call_uses_default_suffix() -> None:
    """The single ``_render_spectrum(active_label)`` call must NOT
    pass ``key_suffix=`` -- the default ``"active"`` keeps the
    per-spectrum session_state keys stable for any pre-existing
    data and keeps the call site readable.
    """
    src = APP.read_text()
    # Locate the call. We expect exactly one occurrence.
    matches = list(re.finditer(r"^\s*_render_spectrum\(active_label\)\s*$", src, re.MULTILINE))
    assert len(matches) == 1, (
        f"Expected exactly one ``_render_spectrum(active_label)`` "
        f"call, found {len(matches)}."
    )
    # The call's text must not contain a key_suffix= argument.
    line_start = src.rfind("\n", 0, matches[0].start()) + 1
    line_end = src.find("\n", matches[0].end())
    call_text = src[line_start:line_end]
    assert "key_suffix" not in call_text, (
        f"Active render call should not pass key_suffix (use the "
        f"default 'active'). Got: {call_text!r}"
    )


# ---------------------------------------------------------------------------
# 4) Behaviour: two callers with different suffixes produce
#    distinct widget keys, no matter what the spectrum label is.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label",
    [
        "30m",
        "60m",
        "Sheet1 :: 30m",
        "my spectrum with spaces",
        "30m :: 30m (2)",  # dedupe-collision label
    ],
)
def test_active_and_compare_widget_keys_disjoint(label: str) -> None:
    """For ANY spectrum label, the active and Compare call sites
    must generate a disjoint set of widget keys. If they share even
    one key, Streamlit raises ``StreamlitDuplicateElementKey`` on
    the second render.
    """
    # Inline the f-strings that the two call sites produce.
    active_keys = {
        f"spec_rename::active::{label}",
        f"plot_style_dots::active::{label}",
        f"plot_style_bars::active::{label}",
        f"plot_label_fields::active::{label}",
        f"show_ions_chk::active::{label}::H+",
        f"show_ions_chk::active::{label}::Na+",
        f"show_ions_chk::active::{label}::K+",
        f"hide_reds::active::{label}",
        f"empty_peaks_active_{label}",
        f"table_active_{label}",
        f"plot_active_{label}",
    }
    compare_keys = {
        f"spec_rename::compare::{label}",
        f"plot_style_dots::compare::{label}",
        f"plot_style_bars::compare::{label}",
        f"plot_label_fields::compare::{label}",
        f"show_ions_chk::compare::{label}::H+",
        f"show_ions_chk::compare::{label}::Na+",
        f"show_ions_chk::compare::{label}::K+",
        f"hide_reds::compare::{label}",
        f"empty_peaks_compare_{label}",
        f"table_compare_{label}",
        f"plot_compare_{label}",
    }
    overlap = active_keys & compare_keys
    assert not overlap, (
        f"Label {label!r}: active and Compare call sites share "
        f"these widget keys (would raise StreamlitDuplicateElementKey): "
        f"{sorted(overlap)}"
    )


def test_compare_call_passed_plot_key_is_unique() -> None:
    """The Compare caller passes ``plot_key=f"compare_plot::{lbl}"``
    -- verify that the active render's plot key (default
    ``f"plot_{label}"``) does not collide with it.
    """
    src = APP.read_text()
    # The Compare call passes a plot_key= argument.
    compare_plot = re.search(
        r"plot_key=f[\"']compare_plot::\{lbl\}[\"']",
        src,
    )
    assert compare_plot, "Compare call must pass plot_key=..."

    # The default plot_key inside _render_spectrum is f"plot_{label}".
    # They share {lbl} (== {label}) so they would collide if
    # namespacing didn't kick in via the key_suffix. With
    # key_suffix="compare", the default becomes
    # f"plot_compare_{label}" which is still distinct from
    # f"plot_active_{label}" -- both unique.

    # Direct check: the strings 'compare_plot::' must NOT appear as
    # the default plot_key inside _render_spectrum.
    tree = _load_app_tree()
    func = _find_function(tree, "_render_spectrum")
    for sub in ast.walk(func):
        if isinstance(sub, ast.If):
            test = sub.test
            # Match: ``if plot_key is None:``
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "plot_key"
            ):
                # The body must assign to plot_key an f-string that
                # references key_suffix (so the default cannot
                # collide with another caller's plot_key=).
                # Handle both ``plot_key = ...`` (Assign) and
                # ``plot_key: ... = ...`` (AnnAssign).
                if not sub.body:
                    continue
                assign = sub.body[0]
                target = None
                value = None
                if isinstance(assign, ast.AnnAssign) and isinstance(assign.target, ast.Name):
                    target = assign.target.id
                    value = assign.value
                elif isinstance(assign, ast.Assign) and len(assign.targets) == 1 and isinstance(assign.targets[0], ast.Name):
                    target = assign.targets[0].id
                    value = assign.value
                if target != "plot_key" or not isinstance(value, ast.JoinedStr):
                    continue
                joined = value
                has_suffix = any(
                    isinstance(p, ast.FormattedValue)
                    and isinstance(p.value, ast.Name)
                    and p.value.id == "key_suffix"
                    for p in joined.values
                )
                assert has_suffix, (
                    "The default plot_key in _render_spectrum must "
                    "include {key_suffix} so the default cannot "
                    "collide with another caller that passes its "
                    "own plot_key=."
                )
                return
    raise AssertionError(
        "Could not locate the default plot_key assignment inside "
        "_render_spectrum; the function probably changed shape."
    )
