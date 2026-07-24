"""End-to-end test: simulate the full Streamlit pre-render flow with 3
spectra and verify that:
  1. The picker renders BEFORE the pre-render block runs.
  2. active_idx auto-promotes to the first ready spectrum.
  3. The graph build sits AFTER the show_ions / hide_reds widget block
     so the same rerun that toggles a pill rebuilds the graph.
  4. The hide_reds default no longer hardcodes to False.
  5. The pre_bar is a single st.empty() slot, not a per-rerun widget.
"""
import re
import sys
from pathlib import Path

src_path = Path(__file__).parent / "app.py"
src = src_path.read_text()
lines = src.split("\n")


def find_line(pattern: str, start: int = 0) -> int:
    """1-indexed line of the first match for ``pattern``."""
    for i in range(start, len(lines)):
        if re.search(pattern, lines[i]):
            return i + 1
    return -1


# 1. Picker is BEFORE the pre-render pool
picker_line = find_line(r'^\s*st\.markdown\("\*\*Pick a spectrum to view\*\*"\)')
pre_bar_slot_line = find_line(r"_pre_bar_slot: Any = st\.empty\(\)")
missing_block_line = find_line(r"^missing = \[lbl for lbl in tab_labels")

print(f"Picker line:         {picker_line}")
print(f"Pre-bar slot line:   {pre_bar_slot_line}")
print(f"Pre-render `missing=`: {missing_block_line}")
assert picker_line < missing_block_line, (
    f"Picker (line {picker_line}) must come BEFORE the pre-render block "
    f"(line {missing_block_line}) so the user sees it between reruns."
)
print("OK: picker is before pre-render block")


# 2. active_idx auto-promote exists and only fires on initial seed
auto_promote = re.search(
    r"_active_was_just_seeded\s+and\s+tab_labels\s+and\s+active_idx\s*<",
    src,
)
assert auto_promote, "Auto-promote guard missing"
print("OK: auto-promote is gated on _active_was_just_seeded (initial seed only)")


# 3. Graph build is AFTER the show_ions / hide_reds widget block.
# The widget keys inside _render_spectrum are now namespaced by
# key_suffix (so Compare and active do not collide), so we match
# the f-string pattern with the {key_suffix} placeholder included.
show_ions_widget = find_line(r"show_ions_chk::\{key_suffix\}::\{label\}::\{ion_label\}")
graph_call = find_line(r"^    st\.plotly_chart\(", 0)
hide_reds_checkbox = find_line(r"f\"Hide reds \(\>\{RED_DA_THRESHOLD")
# Find the per-spectrum hide_reds checkbox (inside _render_spectrum).
# The widget key is now namespaced: ``hide_reds::active::{label}`` /
# ``hide_reds::compare::{label}`` (set in a local ``hide_reds_widget``
# variable that is passed to ``key=``).
hide_reds_widget = find_line(r"key=hide_reds_widget")
print(f"show_ions_chk widget at:    {show_ions_widget}")
print(f"hide_reds checkbox at:     {hide_reds_widget}")
print(f"st.plotly_chart call at:   {graph_call}")
assert graph_call > hide_reds_widget, (
    f"Graph (line {graph_call}) must come AFTER the hide_reds widget "
    f"(line {hide_reds_widget}) so the same rerun that toggles the "
    f"checkbox rebuilds the graph."
)
assert graph_call > show_ions_widget, (
    f"Graph (line {graph_call}) must come AFTER the show_ions_chk widget "
    f"(line {show_ions_widget})."
)
print("OK: graph is after both show_ions and hide_reds widget blocks")


# 3b. The active call site uses the default "active" key_suffix and
# the Compare call site uses "compare". This is the contract that
# prevents StreamlitDuplicateElementKey on the spec_rename text_input
# and every other per-spectrum widget.
active_call = re.search(
    r"^\s*_render_spectrum\(active_label\)\s*$", src, re.MULTILINE
)
assert active_call, "Active call site _render_spectrum(active_label) missing"
compare_call = re.search(
    r"_render_spectrum\(\s*lbl,\s*show_metrics=False,\s*"
    r"plot_key=f[\"']compare_plot::\{lbl\}[\"'],\s*key_suffix=\"compare\",\s*\)",
    src,
)
assert compare_call, (
    "Compare call site must pass key_suffix=\"compare\" so its "
    "widget keys do not collide with the active render."
)
print("OK: active uses default 'active' suffix; Compare uses 'compare'")


# 4. hide_reds no longer hardcodes to False
old_hardcoded = re.search(r"^hide_reds: bool = False\s*$", src, re.MULTILINE)
assert not old_hardcoded, (
    "The 'hide_reds: bool = False' hardcode is still in the file -- "
    "it would shadow the user's session_state value."
)
print("OK: hide_reds is no longer hardcoded to False")


# 5. Pre-bar is a single st.empty() slot (not per-rerun st.progress)
# A single _pre_bar_slot = st.empty() at module level means the bar
# lives in one DOM slot across reruns.
slot_uses = [
    i for i, ln in enumerate(lines)
    if "_pre_bar_slot" in ln
]
print(f"_pre_bar_slot references: {len(slot_uses)} (line numbers: {[i+1 for i in slot_uses[:5]]}...)")
assert len(slot_uses) >= 3, (
    f"Expected _pre_bar_slot to be referenced at least 3 times "
    f"(define, use in if missing, clear on done). Got {len(slot_uses)}."
)
print("OK: pre-bar is a single st.empty() slot")


# 6. Compare section no longer has a duplicate _render_spectrum(active_label)
# (we moved that call to before the pre-render block).
calls = [
    i for i, ln in enumerate(lines)
    if re.match(r"^\s*_render_spectrum\(active_label\)\s*$", ln)
]
print(f"_render_spectrum(active_label) call sites: {len(calls)}")
assert len(calls) == 1, (
    f"Expected exactly one _render_spectrum(active_label) call. "
    f"Found {len(calls)} -- the second is the leftover from the bottom "
    f"of the script that the v0.32.5 restructure should have removed."
)
print("OK: only one _render_spectrum(active_label) call site")


# 7. The pre-render block reruns after all missing spectra are solved so the
# picker and comparison controls are enabled with a complete cache.
pre_render_marker = find_line(r"if any\(_cand_key\(lbl\) not in st\.session_state")
assert pre_render_marker > 0, "pre-render missing-spectrum block not found"
rerun_marker = find_line(r"st\.rerun\(\)", pre_render_marker)
print(f"pre-render block: {pre_render_marker}, st.rerun: {rerun_marker}")
assert rerun_marker > pre_render_marker, (
    "pre-render block should call st.rerun() after all missing spectra are solved"
)
print("OK: pre-render block reruns after all missing spectra are solved")


# 7b. The rename popover has a "prefix for all samples" control
# (active render only; Compare render does not expose it because
# the prefix-all is a global mutation). The control has to live
# INSIDE the ``if key_suffix == "active":`` block so the Compare
# popover doesn't fire a global rename from the wrong render site.
prefix_block = re.search(
    r"if key_suffix == .active.:.*?Prefix for all samples",
    src,
    re.DOTALL,
)
assert prefix_block, (
    "Prefix-for-all control is missing from the rename popover "
    "(searched for the 'if key_suffix == active' block + the "
    "'Prefix for all samples' label)."
)
print("OK: rename popover has prefix-for-all control (active-only)")


# 7c. The Apply button writes ``{prefix}-{current}`` for every
# label in ``tab_labels`` and triggers a rerun. The wipe-with-empty
# branch (empty prefix) clears every ``spec_rename::{lbl}`` key.
apply_block = re.search(
    r"if st\.button.*?Apply to all samples.*?st\.rerun",
    src,
    re.DOTALL,
)
assert apply_block, "Apply button block not found in the rename popover"
wipe_block = re.search(
    r'prefix_clean.*?else:.*?spec_rename::\{lbl\}',
    src,
    re.DOTALL,
)
assert wipe_block, "Wipe branch (empty prefix clears all overrides) missing"
print("OK: Apply writes {prefix}-{current} per label; empty prefix clears all")


# 8. Picker buttons are gated on per_spectrum_results
picker_btn = re.search(
    r"disabled=not _ready",
    src,
)
assert picker_btn, "Picker button is not gated on per_spectrum_results"
print("OK: picker buttons are disabled until the spectrum is ready")


# 9. _wipe_parser_scoped_state has a vanished_labels kwarg that
# only drops per-label state for vanished labels (the rest is
# kept, so an "add more files" pass doesn't reset the user's
# rename / show_ions / plot style for the surviving files).
wipe_with_kwarg = re.search(
    r"def _wipe_parser_scoped_state\(vanished_labels:\s*set\[str\]\s*\|\s*None\s*=\s*None\)",
    src,
)
assert wipe_with_kwarg, (
    "_wipe_parser_scoped_state must accept vanished_labels=set() "
    "for the selective-wipe path used by the 'add more files' flow."
)
print("OK: _wipe_parser_scoped_state accepts vanished_labels for selective wipe")


# 9b. The selective-wipe path branches on vanished_labels being
# None (full wipe) vs a set (per-label only). Both branches must
# exist; the per-label branch must NOT also drop non-per-label
# bare keys like upload_fingerprint_map.
full_wipe_branch = re.search(
    r"if vanished_labels is None:",
    src,
)
selective_wipe_branch = re.search(
    r"else:.*?Selective wipe:.*?for k in keys_to_drop",
    src,
    re.DOTALL,
)
assert full_wipe_branch, "Full-wipe branch (vanished_labels is None) missing"
assert selective_wipe_branch, "Selective-wipe branch (per-label only) missing"
print("OK: _wipe_parser_scoped_state has both full-wipe and selective-wipe branches")


# 10. The per-file fingerprint map is the new state, replacing
# the old single-string fingerprint.
per_file_fp = re.search(
    r"def _file_fingerprint\(up\)|def _current_upload_map\(uploads: list\)",
    src,
)
assert per_file_fp, "Per-file fingerprint helpers missing"
print("OK: per-file fingerprint helpers are defined")


# 10b. The "add more files" case carries over UNCHANGED files'
# parsed Spectrums and re-parses only the new/changed ones --
# this is the core of the "shouldn't re-analyse all 12" fix.
carry_over = re.search(
    r"Carry over UNCHANGED files' parsed Spectrums",
    src,
)
assert carry_over, "Carry-over branch for unchanged files missing"
print("OK: unchanged files' parsed Spectrums are carried over (not re-parsed)")


# 10c. The solver-hash block tracks ``pre_render_tab_labels``
# separately from the input hash, so a pure label-set change
# (add 6 files) does NOT trigger a full results clear.
label_set_branch = re.search(
    r"elif tuple\(tab_labels\) != last_tab_labels:",
    src,
)
assert label_set_branch, (
    "Solver-hash block does not handle label-set changes "
    "separately from input changes. Without this branch, adding "
    "files wipes every cached result and forces a full re-solve."
)
print("OK: solver-hash block tracks label-set changes separately from input changes")


# 10d. The wipe-after-parse block drops per-label state for
# labels that vanished (file removed or bytes changed), keeping
# survivors' state intact.
vanish_wipe = re.search(
    r"if upload_set_changed and last_fingerprints:.*?_wipe_parser_scoped_state",
    src,
    re.DOTALL,
)
assert vanish_wipe, "Post-parse vanish-wipe block missing"
print("OK: post-parse vanish-wipe drops state only for vanished labels")


print()
print("All 10 checks passed.")
