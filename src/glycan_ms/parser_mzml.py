"""mzML parser for the glycan mass spectrometry analyzer.

Reads a single mzML file (XML, HUPO-PSI standard) using SAX and yields
``Spectrum`` objects populated with ``Peak`` instances from the project's
``core`` module. The decode path is shared with the mzXML parser
(``parser_mzxml._decode_peaks``) -- the same base64 / optional-zlib /
struct-unpack algorithm.

Schema notes (mzML 1.1.0):
  * Root: ``<mzML>`` or ``<indexedmzML>``
  * Spectra live in ``<spectrumList><spectrum>...</spectrum></spectrumList>``
  * MS level, scan id, etc. are encoded as ``<cvParam>`` children of
    ``<spectrum>`` with PSI-MS controlled-vocabulary accessions:
        MS:1000511 = ms level
        MS:1000768 = native spectrum id (preferred scan id)
        MS:1000778 = scan start time (minutes)
  * The peak binary data is in ``<binaryDataArrayList><binaryDataArray>``
    with one ``<cvParam>`` per array:
        MS:1000523 = 32-bit float
        MS:1000521 = 64-bit float
        MS:1000574 = zlib compression
        MS:1000576 = no compression
    plus a ``MS:1000525`` (``MS:1000521``/``MS:1000523``) cvParam on the
    binary data array itself identifying the m/z vs intensity axis.
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
from .parser_mzxml import _StopParsing


# HUPO-PSI controlled-vocabulary accessions used by mzML
_CV_MS_LEVEL = "MS:1000511"
_CV_NATIVE_ID = "MS:1000768"
_CV_SCAN_TIME = "MS:1000018"  # alternative id for retention time
_CV_32BIT = "MS:1000523"
_CV_64BIT = "MS:1000521"
_CV_ZLIB = "MS:1000574"
_CV_NO_COMPRESSION = "MS:1000576"
_CV_MZ_ARRAY = "MS:1000514"
_CV_INTENSITY_ARRAY = "MS:1000515"

_RETENTION_TIME_PATTERN = re.compile(r"^PT(?:(\d*\.?\d*)M)?(?:(\d*\.?\d*)S)?$")


def _convert_retention_time(value: str) -> Optional[float]:
    """Convert an ISO-8601 duration (``PT123.4S``) to seconds."""
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


class _MzMLSAXHandler(xml.sax.handler.ContentHandler):
    """SAX handler that collects every MS1 spectrum in the document.

    The handler tracks three pieces of state:
      * which ``<spectrum>`` we are currently in (and whether it is MS1)
      * the binary data arrays, indexed by m/z vs intensity axis
      * the cvParams that tell us the precision and compression in use
    """

    def __init__(self, *, only_first_ms1: bool, mz_hi: float | None = None) -> None:
        self._only_first_ms1 = only_first_ms1
        self._mz_hi = mz_hi
        self._stopped = False

        self.spectra: List[Spectrum] = []

        # Current spectrum state
        self._in_spectrum = False
        self._is_ms1 = False
        self._native_id: Optional[str] = None

        # Binary data array state. Each array has:
        #   - its own m/z vs intensity axis tag
        #   - cvParams for precision + compression
        #   - the accumulating <binary> chunk
        self._in_binary_data_array = False
        self._in_binary = False
        self._binary_chunks: List[str] = []
        # Per-array state
        self._array_axis: Optional[str] = None  # "mz" or "intensity"
        self._array_precision: int = 64
        self._array_compression: Optional[str] = None
        # Accumulated arrays for the current spectrum
        self._mz_payload: Optional[tuple[list[float], int, Optional[str]]] = None
        self._int_payload: Optional[tuple[list[float], int, Optional[str]]] = None

    # ---- SAX callbacks -------------------------------------------------

    def startElement(self, name: str, attrs) -> None:  # type: ignore[override]
        if self._stopped:
            return

        if name == "spectrum":
            self._in_spectrum = True
            self._is_ms1 = False
            self._native_id = attrs.get("id")
            self._mz_payload = None
            self._int_payload = None

        elif name == "cvParam" and self._in_spectrum:
            accession = attrs.get("accession", "")
            value = attrs.get("value", "")
            if accession == _CV_MS_LEVEL:
                try:
                    self._is_ms1 = int(value) == 1
                except (TypeError, ValueError):
                    self._is_ms1 = False
            elif accession == _CV_NATIVE_ID:
                # <spectrum id="..."> is canonical but the cvParam wins if present.
                if value:
                    self._native_id = value
            elif accession == _CV_32BIT and self._in_binary_data_array:
                self._array_precision = 32
            elif accession == _CV_64BIT and self._in_binary_data_array:
                self._array_precision = 64
            elif accession == _CV_ZLIB and self._in_binary_data_array:
                self._array_compression = "zlib"
            elif accession == _CV_NO_COMPRESSION and self._in_binary_data_array:
                self._array_compression = None
            elif accession == _CV_MZ_ARRAY and self._in_binary_data_array:
                self._array_axis = "mz"
            elif accession == _CV_INTENSITY_ARRAY and self._in_binary_data_array:
                self._array_axis = "intensity"

        elif name == "binaryDataArray" and self._in_spectrum and self._is_ms1:
            self._in_binary_data_array = True
            self._array_axis = None
            self._array_precision = 64
            self._array_compression = None

        elif name == "binary" and self._in_binary_data_array and self._is_ms1:
            self._in_binary = True
            self._binary_chunks = []

    def endElement(self, name: str) -> None:  # type: ignore[override]
        if self._stopped:
            return

        if name == "binary" and self._in_binary:
            self._in_binary = False
            payload = (
                "".join(self._binary_chunks),
                self._array_precision,
                self._array_compression,
            )
            if self._array_axis == "mz":
                self._mz_payload = payload
            elif self._array_axis == "intensity":
                self._int_payload = payload

        elif name == "binaryDataArray" and self._in_binary_data_array:
            self._in_binary_data_array = False
            self._array_axis = None

        elif name == "spectrum" and self._in_spectrum:
            self._in_spectrum = False
            if self._is_ms1 and self._mz_payload is not None and self._int_payload is not None:
                mz_list = _decode_binary_array(*self._mz_payload)
                int_list = _decode_binary_array(*self._int_payload)
                n = min(len(mz_list), len(int_list))
                scan_id = self._parse_scan_id(self._native_id)
                peaks = [
                    Peak(mz=mz_list[i], intensity=int_list[i], scan_id=scan_id)
                    for i in range(n)
                ]
                self.spectra.append(Spectrum(peaks=peaks, mz_hi=self._mz_hi))
                if self._only_first_ms1 and self.spectra:
                    self._stopped = True
                    raise _StopParsing()

    def characters(self, content: str) -> None:  # type: ignore[override]
        if self._in_binary:
            self._binary_chunks.append(content)

    # ---- Helpers -------------------------------------------------------

    @staticmethod
    def _parse_scan_id(native_id: Optional[str]) -> Optional[int]:
        """Best-effort integer scan id from a native spectrum id string.

        PSI-MS allows free-form native ids; the common conventions are
        ``scan=42`` (mzML 1.1) or a bare integer. We try both.
        """
        if not native_id:
            return None
        m = re.search(r"scan=(\d+)", native_id)
        if m:
            return int(m.group(1))
        try:
            return int(native_id)
        except (TypeError, ValueError):
            return None


def _decode_binary_array(
    raw: str, precision: int, compression: Optional[str]
) -> List[float]:
    """Decode one mzML ``<binaryDataArray>`` payload to a list of floats.

    Unlike mzXML ``<peaks>`` (which interleaves (m/z, intensity) pairs),
    an mzML binary data array is a single float array. The path is the
    same otherwise: base64 -> optional zlib -> struct unpack.
    """
    if not raw:
        return []
    data = base64.b64decode(raw)
    if compression and compression != "none":
        data = zlib.decompress(data)
    endian = "!"  # mzML is always network byte order
    fmt_char = "d" if precision == 64 else "f"
    item_size = struct.calcsize(endian + fmt_char)
    if item_size == 0 or len(data) % item_size != 0:
        return []
    count = len(data) // item_size
    return list(struct.unpack(endian + fmt_char * count, data))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_mzml(
    path: str, *, only_first_ms1: bool = False, mz_hi: float | None = None
) -> List[Spectrum]:
    """Parse ``path`` and return one ``Spectrum`` per MS1 spectrum.

    Non-MS1 spectra (MS2, MS3, ...) are skipped; they aren't useful
    for the glycan composition solver in this project. When
    ``only_first_ms1`` is True the SAX stream is aborted as soon as
    the first MS1 spectrum has been decoded.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    if not os.path.exists(path):
        raise FileNotFoundError("mzML file not found: " + path)

    handler = _MzMLSAXHandler(only_first_ms1=only_first_ms1, mz_hi=mz_hi)
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    with open(path, "rb") as fh:
        try:
            parser.parse(fh)
        except _StopParsing:
            pass
        except xml.sax.SAXException as exc:
            raise RuntimeError("Failed to parse mzML: " + path) from exc

    return handler.spectra


def first_scan_spectrum_mzml(
    path: str, *, mz_hi: float | None = None
) -> Spectrum:
    """Return only the first MS1 spectrum in ``path``.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    spectra = parse_mzml(path, only_first_ms1=True, mz_hi=mz_hi)
    if not spectra:
        raise ValueError("No MS1 spectra found in mzML file: " + path)
    return spectra[0]


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def _main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python parser_mzml.py path/to/file.mzML", file=sys.stderr)
        return 1

    path = argv[1]
    spectra = parse_mzml(path)
    print(f"{len(spectra)} spectra")

    if spectra:
        first = spectra[0]
        print(f"first spectrum: {len(first.peaks)} peaks")
        for peak in first.peaks[:5]:
            print(f"  m/z={peak.mz:.4f}  intensity={peak.intensity:.2e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
