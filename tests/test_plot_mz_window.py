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


def test_spectrum_plot_uses_observed_extent_without_empty_prefix() -> None:
    """A 600-5000 selection must start at its first observed peak."""
    import pandas as pd

    from app import _spectrum_plot
    from glycan_ms.core import Peak

    figure = _spectrum_plot(
        [
            Peak(mz=400.0, intensity=10.0),
            Peak(mz=750.0, intensity=20.0),
            Peak(mz=4200.0, intensity=25.0),
            Peak(mz=5200.0, intensity=30.0),
        ],
        pd.DataFrame(),
        "sample",
        mz_min=600.0,
        mz_max=5000.0,
    )

    assert tuple(figure.layout.xaxis.range) == (750.0, 4200.0)
    plotted_mz = [
        value
        for trace in figure.data
        if trace.name == "Spectrum"
        for value in trace.x
    ]
    assert plotted_mz == [750.0, 4200.0]
    spectrum = next(trace for trace in figure.data if trace.name == "Spectrum")
    assert spectrum.hoverinfo == "x+y"
    assert len(spectrum.customdata) == 2
    noise_x = [
        value
        for trace in figure.data
        if str(trace.name).startswith("Noise floor")
        for value in trace.x
    ]
    assert min(noise_x) == 750.0
    assert max(noise_x) == 4200.0


def test_auto_zoom_focuses_1000_to_5050_and_uses_detail_y_scale() -> None:
    import pandas as pd

    from app import _spectrum_plot
    from glycan_ms.core import Peak

    peaks = [
        Peak(mz=700.0, intensity=2_000_000.0),
        *[
            Peak(mz=1100.0 + index * 150.0, intensity=100.0 + index)
            for index in range(20)
        ],
        Peak(mz=1500.0, intensity=1_000_000.0),
        Peak(mz=5100.0, intensity=80.0),
    ]
    figure = _spectrum_plot(
        peaks,
        pd.DataFrame(),
        "sample",
        mz_min=600.0,
        mz_max=6000.0,
        auto_zoom_detail=True,
    )

    spectrum = next(trace for trace in figure.data if trace.name == "Spectrum")
    assert min(spectrum.x) >= 1000.0
    assert max(spectrum.x) <= 5050.0
    assert tuple(figure.layout.xaxis.range) == (min(spectrum.x), max(spectrum.x))
    assert figure.layout.yaxis.range[1] < 1_000_000.0
    noise_x = [
        value
        for trace in figure.data
        if str(trace.name).startswith("Noise floor")
        for value in trace.x
    ]
    assert max(noise_x) <= 5050.0


def test_spectrum_plot_keeps_only_highest_peak_on_same_mz_line() -> None:
    import pandas as pd

    from app import _spectrum_plot
    from glycan_ms.core import Peak

    figure = _spectrum_plot(
        [
            Peak(mz=1200.001, intensity=10.0),
            Peak(mz=1200.004, intensity=90.0),
            Peak(mz=1300.000, intensity=20.0),
        ],
        pd.DataFrame(),
        "sample",
        mz_min=1000.0,
        mz_max=1400.0,
    )

    spectrum = next(trace for trace in figure.data if trace.name == "Spectrum")
    assert list(spectrum.x) == [1200.004, 1300.0]
    assert list(spectrum.y) == [90.0, 20.0]
    assert spectrum.type == "scattergl"


def test_candidate_overlay_keeps_one_highest_point_per_mz_line() -> None:
    import pandas as pd

    from app import _spectrum_plot
    from glycan_ms.core import Peak

    candidates = pd.DataFrame(
        [
            {"mz": 1200.001, "intensity": 40.0, "ion": "H+", "n_galnac": 1, "n_gal": 2, "mz_diff": 0.02},
            {"mz": 1200.003, "intensity": 80.0, "ion": "Na+", "n_galnac": 2, "n_gal": 2, "mz_diff": 0.03},
            {"mz": 1300.000, "intensity": 30.0, "ion": "K+", "n_galnac": 3, "n_gal": 1, "mz_diff": 0.01},
        ]
    )
    figure = _spectrum_plot(
        [Peak(mz=1200.003, intensity=80.0), Peak(mz=1300.0, intensity=30.0)],
        candidates,
        "sample",
        mz_min=1000.0,
        mz_max=1400.0,
    )

    candidate_points = [
        (float(x), float(y))
        for trace in figure.data
        if trace.name in {"H+", "Na+", "K+"}
        for x, y in zip(trace.x, trace.y)
    ]
    assert sorted(candidate_points) == [(1200.003, 80.0), (1300.0, 30.0)]
    assert all(
        trace.type == "scattergl"
        for trace in figure.data
        if trace.name in {"H+", "Na+", "K+"}
    )


def test_plotly_chart_enables_point_selection() -> None:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    render = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_render_spectrum"
    )
    chart_call = next(
        node
        for node in ast.walk(render)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "plotly_chart"
    )
    keywords = {keyword.arg: keyword.value for keyword in chart_call.keywords}

    assert ast.literal_eval(keywords["on_select"]) == "rerun"
    assert ast.literal_eval(keywords["selection_mode"]) == "points"
