import io
import math
from pathlib import Path

from PIL import Image

from app import _compose_graph_grid_png


def _solid_png(color: str, size: tuple[int, int] = (120, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_composer_accepts_arbitrary_graph_counts() -> None:
    graph_count = 17
    payload = _compose_graph_grid_png(
        [_solid_png("navy")] * graph_count,
        "Seventeen graphs",
    )

    with Image.open(io.BytesIO(payload)) as image:
        assert image.format == "PNG"
        expected_columns = math.ceil(math.sqrt(graph_count))
        expected_rows = math.ceil(graph_count / expected_columns)
        assert image.width == 48 + expected_columns * 120 + (expected_columns - 1) * 24
        assert image.height > expected_rows * 80


def test_composer_uses_near_square_grid_for_five_graphs() -> None:
    payload = _compose_graph_grid_png(
        [_solid_png("navy")] * 5,
        "Five graphs",
    )

    with Image.open(io.BytesIO(payload)) as image:
        three_column_width = 48 + 3 * 120 + 2 * 24
        assert image.width == three_column_width
        assert image.height < image.width


def test_project_requires_sized_default_font_support() -> None:
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"

    assert '"pillow>=10.1"' in project_file.read_text(encoding="utf-8")


def test_large_composite_stays_within_bounded_canvas() -> None:
    payload = _compose_graph_grid_png(
        [_solid_png("navy", size=(1200, 750))] * 100,
        "One hundred graphs",
    )

    with Image.open(io.BytesIO(payload)) as image:
        assert image.width <= 6000
        assert image.height <= 6000
