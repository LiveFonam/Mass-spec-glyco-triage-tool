"""Regression tests for keeping the spectrum graph aligned with the m/z window."""

from __future__ import annotations

import ast
from pathlib import Path


APP = Path(__file__).resolve().parent.parent / "app.py"


def test_render_spectrum_passes_sidebar_window_to_plot() -> None:
    """The graph must use both sidebar limits instead of Plotly autorange."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    render = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_render_spectrum"
    )
    plot_call = next(
        node
        for node in ast.walk(render)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_spectrum_plot"
    )
    mz_min = next(
        (keyword.value for keyword in plot_call.keywords if keyword.arg == "mz_min"),
        None,
    )
    mz_max = next(
        (keyword.value for keyword in plot_call.keywords if keyword.arg == "mz_max"),
        None,
    )

    assert mz_min is not None, "_render_spectrum must pass mz_min to the graph"
    assert mz_max is not None, "_render_spectrum must pass mz_max to the graph"
    assert "sidebar_mz_lo" in ast.unparse(mz_min)
    assert "sidebar_mz_hi" in ast.unparse(mz_max)


def test_spectrum_plot_pins_x_axis_to_selected_window() -> None:
    """A 600-5000 selection must not autorange toward zero."""
    import pandas as pd

    from app import _spectrum_plot
    from glycan_ms.core import Peak

    figure = _spectrum_plot(
        [
            Peak(mz=400.0, intensity=10.0),
            Peak(mz=750.0, intensity=20.0),
            Peak(mz=5200.0, intensity=30.0),
        ],
        pd.DataFrame(),
        "sample",
        mz_min=600.0,
        mz_max=5000.0,
    )

    assert tuple(figure.layout.xaxis.range) == (600.0, 5000.0)
    plotted_mz = [
        value
        for trace in figure.data
        if trace.name == "Spectrum"
        for value in trace.x
    ]
    assert plotted_mz == [750.0]
