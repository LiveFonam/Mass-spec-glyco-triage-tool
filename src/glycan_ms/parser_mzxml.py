"""mzXML parser for the glycan mass spectrometry analyzer.

Reads a single mzXML file (SAX streaming) and yields ``Spectrum`` objects
populated with ``Peak`` instances from the project's ``core`` module.

The SAX approach is adapted from mMass's ``mspy.parser_mzxml`` module --
the relevant pieces are:
  * scan hierarchy (parent / child scans)
  * retention time pattern ``^PT(\\d*\\.?\\d*M)?(\\d*\\.?\\d*S)?$``
  * base64 + optional zlib + struct unpack (32-bit 'f' default, 64-bit 'd')
  * endianness from the ``byteOrder`` attribute (``network`` = ``!``,
    ``little`` = ``<``, ``big`` = ``>``)
  * a "stop parsing" sentinel exception so ``first_scan_spectrum`` can
    bail out of the stream as soon as the first MS1 scan is decoded

Only the standard library is required (``xml.sax``, ``base64``, ``zlib``,
``struct``); ``numpy`` is optional and only used to reshape the decoded
peaks into an (N, 2) array for the hot path.
"""

from __future__ import annotations

import base64
import os.path
import re
import struct
import sys
import xml.sax
import zlib
from typing import List, Optional

from .core import Peak, Spectrum


# retention time strings look like "PT123.4S" or "PT1.5M30.0S"
_RETENTION_TIME_PATTERN = re.compile(r"^PT(?:(\d*\.?\d*)M)?(?:(\d*\.?\d*)S)?$")

# Network byte order ("!") is big-endian with the standard struct sizes.
_ENDIAN_MAP = {
    "network": "!",
    "big": ">",
    "little": "<",
}


def _convert_retention_time(value: str) -> Optional[float]:
    """Convert an ISO-8601 duration ("PT...M...S") to seconds."""
    match = _RETENTION_TIME_PATTERN.match(value)
    if not match:
        return None
    minutes = match.group(1)
    seconds = match.group(2)
    total = 0.0
    if minutes:
        total += float(minutes) * 60.0
    if seconds:
        total += float(seconds)
    return total


def _decode_peaks(
    raw: str,
    *,
    byte_order: str,
    compression: Optional[str],
    precision: int,
) -> List[Peak]:
    """Decode a base64 (and possibly zlib-compressed) ``<peaks>`` payload.

    The payload is a flat sequence of (m/z, intensity) pairs.
    """
    if not raw:
        return []

    data = base64.b64decode(raw)
    if compression and compression != "none":
        data = zlib.decompress(data)

    endian = _ENDIAN_MAP.get(byte_order, "!")
    if precision == 64:
        fmt_char = "d"
    else:
        fmt_char = "f"  # 32-bit is the mzXML default

    pair_size = struct.calcsize(endian + fmt_char) * 2
    if pair_size == 0 or len(data) % pair_size != 0:
        # Malformed payload -- return empty rather than crash the whole file.
        return []

    count = len(data) // pair_size
    flat = struct.unpack(endian + fmt_char * (count * 2), data)
    # cast to plain Python floats for the dataclass; cheaper than numpy for
    # the small spectra we typically see in glycan profiling.
    return [
        Peak(mz=float(flat[i * 2]), intensity=float(flat[i * 2 + 1]))
        for i in range(count)
    ]


class _StopParsing(Exception):
    """Internal sentinel: raise to abort the SAX stream early."""


class _MzXMLSAXHandler(xml.sax.handler.ContentHandler):
    """SAX handler that collects every MS1 scan in the document.

    For each ``<scan>`` we grab its attributes (msLevel, retentionTime,
    polarity, etc.), then inside the matching ``<peaks>`` element we
    accumulate the base64 payload and decode it on ``</peaks>``.
    """

    def __init__(self, *, only_first_ms1: bool, mz_hi: float | None = None) -> None:
        self._only_first_ms1 = only_first_ms1
        self._mz_hi = mz_hi
        self._stopped = False

        # Collected spectra (MS1 only). Each entry is a Spectrum.
        self.spectra: List[Spectrum] = []

        # Current scan being built.
        self._scan_id: Optional[int] = None
        self._scan_level: Optional[int] = None
        self._is_ms1 = False

        # Peaks state.
        self._in_peaks = False
        self._peaks_chunks: List[str] = []
        self._peaks_byte_order = "network"
        self._peaks_compression: Optional[str] = None
        self._peaks_precision: int = 32

    # ---- SAX callbacks -------------------------------------------------

    def startElement(self, name: str, attrs) -> None:  # type: ignore[override]
        if self._stopped:
            return

        if name == "scan":
            level = attrs.get("msLevel", "1")
            try:
                level_int = int(level)
            except (TypeError, ValueError):
                level_int = 1
            self._scan_level = level_int
            self._is_ms1 = level_int == 1

            num = attrs.get("num")
            try:
                self._scan_id = int(num) if num is not None else None
            except (TypeError, ValueError):
                self._scan_id = None

        elif name == "peaks" and self._is_ms1:
            self._in_peaks = True
            self._peaks_chunks = []
            self._peaks_byte_order = attrs.get("byteOrder", "network")
            compression = attrs.get("compressionType")
            self._peaks_compression = compression if compression else None
            precision_attr = attrs.get("precision", "32")
            try:
                self._peaks_precision = int(precision_attr)
            except (TypeError, ValueError):
                self._peaks_precision = 32

    def endElement(self, name: str) -> None:  # type: ignore[override]
        if self._stopped:
            return

        if name == "peaks" and self._in_peaks:
            self._in_peaks = False
            raw = "".join(self._peaks_chunks)
            peaks = _decode_peaks(
                raw,
                byte_order=self._peaks_byte_order,
                compression=self._peaks_compression,
                precision=self._peaks_precision,
            )
            # Stamp each peak with its scan_id so downstream code can
            # trace a candidate back to its source spectrum.
            scan_id = self._scan_id
            if scan_id is not None:
                peaks = [
                    Peak(mz=p.mz, intensity=p.intensity, scan_id=scan_id)
                    for p in peaks
                ]
            self.spectra.append(Spectrum(peaks=peaks, mz_hi=self._mz_hi))

            if self._only_first_ms1 and self.spectra:
                self._stopped = True
                raise _StopParsing()

        elif name == "scan":
            self._scan_id = None
            self._scan_level = None
            self._is_ms1 = False

    def characters(self, content: str) -> None:  # type: ignore[override]
        if self._in_peaks:
            self._peaks_chunks.append(content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_mzxml(
    path: str, *, only_first_ms1: bool = False, mz_hi: float | None = None
) -> List[Spectrum]:
    """Parse ``path`` and return one ``Spectrum`` per MS1 scan.

    Non-MS1 scans (MS2, MS3, ...) are skipped; they aren't useful for
    the glycan composition solver in this project. When
    ``only_first_ms1`` is True the SAX stream is aborted as soon as
    the first MS1 scan has been decoded.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("mzXML file not found: " + path)

    handler = _MzXMLSAXHandler(only_first_ms1=only_first_ms1, mz_hi=mz_hi)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    with open(path, "rb") as fh:
        try:
            parser.parse(fh)
        except _StopParsing:
            pass
        except xml.sax.SAXException as exc:
            raise RuntimeError("Failed to parse mzXML: " + path) from exc

    return handler.spectra


def first_scan_spectrum(
    path: str, *, mz_hi: float | None = None
) -> Spectrum:
    """Return only the first MS1 spectrum in ``path``.

    Streams the file via SAX and bails out as soon as the first MS1
    scan has been decoded, so the function is fast even for very
    large mzXML files.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    spectra = parse_mzxml(path, only_first_ms1=True, mz_hi=mz_hi)
    if not spectra:
        raise ValueError("No MS1 scans found in mzXML file: " + path)
    return spectra[0]


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python parser_mzxml.py path/to/file.mzXML", file=sys.stderr)
        return 1

    path = argv[1]
    spectra = parse_mzxml(path)
    print(f"{len(spectra)} scans")

    if spectra:
        first = spectra[0]
        print(f"first scan: {len(first.peaks)} peaks")
        for peak in first.peaks[:5]:
            print(f"  m/z={peak.mz:.4f}  intensity={peak.intensity:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
