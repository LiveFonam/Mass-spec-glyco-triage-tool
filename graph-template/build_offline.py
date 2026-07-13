"""Build the fully self-contained graph-template HTML file.

Run from any working directory with ``python graph-template/build_offline.py``
or from this directory with ``python build_offline.py``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "index.html"
OUTPUT = ROOT / "index-offline.html"

VENDOR_FILES = {
    "https://cdn.plot.ly/plotly-3.6.0.min.js": (
        ROOT / "vendor" / "plotly-3.6.0.min.js",
        "41a395c2d558d13d3655a1ebafaa67a072c2c1ac8c269e0ee67e18c9a137ac99",
    ),
    "https://cdn.sheetjs.com/xlsx-0.20.3/package/dist/xlsx.full.min.js": (
        ROOT / "vendor" / "xlsx-0.20.3.full.min.js",
        "cc015130aa8521e7f088f88898eba949ccdcbfb38df0bd129b44b7273c3a6f41",
    ),
    "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js": (
        ROOT / "vendor" / "jszip-3.10.1.min.js",
        "acc7e41455a80765b5fd9c7ee1b8078a6d160bbbca455aeae854de65c947d59e",
    ),
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def script_safe(source: str) -> str:
    """Prevent embedded JS text from terminating its containing script tag."""
    return re.sub(r"</script", r"<\\/script", source, flags=re.IGNORECASE)


def replace_once(document: str, old: str, new: str) -> str:
    count = document.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one occurrence, found {count}: {old}")
    return document.replace(old, new, 1)


def verified_vendor(path: Path, expected_sha256: str) -> str:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return payload.decode("utf-8-sig")


def build() -> Path:
    document = read_text(SOURCE)

    stylesheet_tag = '<link rel="stylesheet" href="styles.css">'
    document = replace_once(
        document,
        stylesheet_tag,
        f"<style>\n{read_text(ROOT / 'styles.css')}\n</style>",
    )

    for remote_url, (local_path, checksum) in VENDOR_FILES.items():
        pattern = re.compile(
            rf'<script\s+src="{re.escape(remote_url)}"(?:\s+charset="utf-8")?></script>'
        )
        matches = pattern.findall(document)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one script tag for {remote_url}, found {len(matches)}"
            )
        embedded = f"<script>\n{script_safe(verified_vendor(local_path, checksum))}\n</script>"
        document = pattern.sub(lambda _: embedded, document, count=1)

    document = replace_once(
        document,
        '<script src="app.js"></script>',
        f"<script>\n{script_safe(read_text(ROOT / 'app.js'))}\n</script>",
    )
    document = replace_once(document, ">CDN edition</div>", ">Offline edition</div>")

    external_assets = re.findall(
        r'<(?:script|link)\b[^>]*(?:src|href)=["\']https?://',
        document,
        flags=re.IGNORECASE,
    )
    if external_assets:
        raise RuntimeError("Offline output still contains an external script or stylesheet")

    OUTPUT.write_text(document, encoding="utf-8", newline="\n")
    return OUTPUT


if __name__ == "__main__":
    result = build()
    size_mib = result.stat().st_size / (1024 * 1024)
    print(f"Built {result} ({size_mib:.2f} MiB)")
