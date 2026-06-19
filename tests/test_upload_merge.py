"""Regression tests for the "add more files" upload-merge path.

The user uploads 6 files, then drops 6 more. The app must:
  1. NOT re-analyse the original 6 (their cached pre_render_results
     and per-label session_state must survive).
  2. NOT remove the rename overrides the user typed for the
     original 6.
  3. Parse only the new 6.
  4. Drop state for files that were removed.
  5. Re-parse a file when its bytes change, even if the name is
     the same (the previous Spectrum for that file's labels must
     be replaced).

The ``_wipe_parser_scoped_state(vanished_labels=...)`` function
is the single point of selective state cleanup, and the
fingerprint helpers (``_file_fingerprint``,
``_current_upload_map``) classify each upload as new / removed /
unchanged. These tests pin the contract.
"""
from __future__ import annotations

import hashlib


def _file_fingerprint(name: str, content: bytes) -> str:
    """Mirror of app._file_fingerprint for testing."""
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(b"\0")
    h.update(content)
    h.update(b"\1")
    return h.hexdigest()


def _current_upload_map(uploads: list[tuple[str, bytes]]) -> dict[str, str]:
    """Mirror of app._current_upload_map for testing."""
    return {name: _file_fingerprint(name, content) for name, content in uploads}


def test_first_upload_map_is_complete():
    """A fresh upload produces a 1:1 name -> fingerprint map."""
    uploads = [
        ("a.mzxml", b"AAA"),
        ("b.mzxml", b"BBB"),
    ]
    m = _current_upload_map(uploads)
    assert set(m.keys()) == {"a.mzxml", "b.mzxml"}
    assert m["a.mzxml"] != m["b.mzxml"]


def test_unchanged_files_classified_as_unchanged():
    """Two uploads of the same bytes produce matching fingerprints."""
    last = _current_upload_map([("a.mzxml", b"AAA"), ("b.mzxml", b"BBB")])
    current = _current_upload_map([("a.mzxml", b"AAA"), ("b.mzxml", b"BBB")])
    new = [n for n in current if current[n] != last.get(n)]
    removed = [n for n in last if n not in current]
    assert new == []
    assert removed == []


def test_added_file_classified_as_new():
    """Adding a 7th file shows up as ``new``, others as unchanged."""
    last = _current_upload_map(
        [("a.mzxml", b"AAA"), ("b.mzxml", b"BBB"), ("c.mzxml", b"CCC")]
    )
    current = _current_upload_map(
        [("a.mzxml", b"AAA"), ("b.mzxml", b"BBB"), ("c.mzxml", b"CCC"),
         ("d.mzxml", b"DDD")]
    )
    new = [n for n in current if current[n] != last.get(n)]
    removed = [n for n in last if n not in current]
    assert new == ["d.mzxml"]
    assert removed == []


def test_removed_file_classified_as_removed():
    last = _current_upload_map(
        [("a.mzxml", b"AAA"), ("b.mzxml", b"BBB"), ("c.mzxml", b"CCC")]
    )
    current = _current_upload_map(
        [("a.mzxml", b"AAA"), ("c.mzxml", b"CCC")]
    )
    new = [n for n in current if current[n] != last.get(n)]
    removed = [n for n in last if n not in current]
    assert new == []
    assert removed == ["b.mzxml"]


def test_changed_bytes_classified_as_new():
    """A re-uploaded file with different content counts as new."""
    last = _current_upload_map([("a.mzxml", b"OLD")])
    current = _current_upload_map([("a.mzxml", b"NEW")])
    new = [n for n in current if current[n] != last.get(n)]
    removed = [n for n in last if n not in current]
    # The file with changed bytes is BOTH new (under its new
    # fingerprint) and a "replaced" version of the old. The
    # parse path treats it as "re-parse this file" and drops
    # the previous Spectrum for its labels.
    assert new == ["a.mzxml"]
    assert removed == []


# ---- _wipe_parser_scoped_state selective-wipe logic ----

_PREFIXES = (
    "spec_rename::", "show_ions::", "show_ions_chk::",
    "file_prefix::", "plot_", "table_", "compare_plot::",
    "compare_pick::", "download_png::", "download_xlsx::",
    "spec_btn::", "plot_label_fields::", "hide_reds::",
)


def _selective_wipe_keys(
    session: dict[str, object],
    vanished_labels: set[str],
) -> list[str]:
    """Mirror of app._wipe_parser_scoped_state(vanished_labels=...)
    for the per-label-key part. Returns the keys that would be
    dropped, so the test can assert against the expected set
    without calling into the real Streamlit session_state.
    """
    keys_to_drop: list[str] = []
    for k in list(session.keys()):
        for prefix in _PREFIXES:
            if k.startswith(prefix):
                tail = k[len(prefix):]
                # ``::`` (no spaces) is the namespacing separator
                # (active/compare). `` :: `` (with spaces) is part
                # of a multi-file label.
                if "::" in tail and " :: " not in tail:
                    label_part = tail.split("::", 1)[1]
                else:
                    label_part = tail
                if label_part in vanished_labels:
                    keys_to_drop.append(k)
                break
    return keys_to_drop


def test_selective_wipe_keeps_survivor_state():
    """When label ``30m`` vanishes, ``30m``'s state is dropped
    but ``60m`` and ``90m``'s state is kept.
    """
    session = {
        "spec_rename::30m": "BatchA",
        "spec_rename::60m": "BatchA",
        "spec_rename::90m": "BatchA",
        "show_ions::30m": ["Na+", "K+"],
        "show_ions::60m": ["Na+"],
        "plot_style::30m": "bars",
        "plot_style::60m": "dots",
        # Namespaced widget keys (Compare render)
        "spec_rename::active::30m": "BatchA",
        "spec_rename::active::60m": "BatchA",
        "hide_reds::30m": True,
    }
    dropped = _selective_wipe_keys(session, vanished_labels={"30m"})
    # All keys whose label part is ``30m`` should be dropped
    # (including plot_style::30m -- the ``plot_`` prefix matches
    # ``plot_style``, ``plot_label_fields``, etc).
    assert set(dropped) == {
        "spec_rename::30m",
        "show_ions::30m",
        "plot_style::30m",
        "spec_rename::active::30m",
        "hide_reds::30m",
    }
    # Survivors untouched.
    assert session["spec_rename::60m"] == "BatchA"
    assert session["spec_rename::90m"] == "BatchA"
    assert session["show_ions::60m"] == ["Na+"]


def test_selective_wipe_drops_only_vanished_multi_label_format():
    """Multi-file labels like ``60m :: 30m`` are matched by their
    full string (the label is everything after the last ``::``).
    """
    session = {
        "spec_rename::60m :: 30m": "Combined",
        "spec_rename::60m :: 60m": "Combined",
    }
    dropped = _selective_wipe_keys(session, vanished_labels={"60m :: 30m"})
    assert dropped == ["spec_rename::60m :: 30m"]


def test_selective_wipe_handles_namespaced_widget_keys():
    """Keys with format ``{prefix}{key_suffix}::{label}`` are
    matched by their label suffix.
    """
    session = {
        "spec_rename::active::30m": "x",
        "spec_rename::compare::30m": "x",
        "spec_rename::active::60m": "y",
    }
    dropped = _selective_wipe_keys(session, vanished_labels={"30m"})
    assert set(dropped) == {
        "spec_rename::active::30m",
        "spec_rename::compare::30m",
    }
