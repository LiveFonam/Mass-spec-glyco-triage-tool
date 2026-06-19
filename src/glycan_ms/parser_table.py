"""Tabular peak-list parsers for CSV and Excel inputs.

Reads a two-column (m/z, intensity) peak list from a ``.csv``, ``.xls`` or
``.xlsx`` file and returns a :class:`Spectrum` of :class:`Peak` objects.

Column names are matched case-insensitively against a small set of common
aliases; any NaN/missing row is dropped silently. The first column is
ignored if it looks like a pandas-style index (``Unnamed: 0``).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd

from .core import Peak, Spectrum


# Aliases normalised to lowercase, no whitespace.
_MZ_ALIASES: frozenset[str] = frozenset(
    {
        "mass",
        "mass.",
        "m/z",
        "m/z.",
        "mz",
        "m_z",
        "m z",
        "moverz",
        "mass/charge",
    }
)
_INTENSITY_ALIASES: frozenset[str] = frozenset(
    {
        "intensity",
        "intensity.",
        "intens",
        "intens.",
        "abundance",
        "height",
        "area",
    }
)

# Header row that the parser will recognise as the column-name row of
# a Bruker / Agilent / Waters-style peak list export.
_TIMEPOINT_RE = re.compile(r"^\d+\s*(m|h|s)$", re.IGNORECASE)


def _normalise(name: object) -> str:
    """Lowercase, strip, and collapse ``/`` / space / underscore separators."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("/", "")
    )


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find the m/z and intensity columns in ``df``.

    Raises ``ValueError`` listing the columns seen if either cannot be
    located. The first column is skipped when it looks like a pandas
    index named ``Unnamed: 0``.
    """
    # Build a mapping from normalised name to original column.
    lookup: dict[str, str] = {}
    for col in df.columns:
        norm = _normalise(col)
        if norm:
            lookup[norm] = col

    mz_col = next(
        (lookup[n] for n in _MZ_ALIASES if n in lookup),
        None,
    )
    intensity_col = next(
        (lookup[n] for n in _INTENSITY_ALIASES if n in lookup),
        None,
    )

    if mz_col is None or intensity_col is None:
        seen = ", ".join(repr(c) for c in df.columns)
        raise ValueError(
            f"could not locate m/z and/or intensity columns; saw: {seen}"
        )
    return mz_col, intensity_col


def _row_has_header_values(row: pd.Series) -> bool:
    """Return True if the row contains any string that looks like a known header."""
    for v in row:
        if not isinstance(v, str):
            continue
        norm = _normalise(v)
        if norm in _MZ_ALIASES or norm in _INTENSITY_ALIASES:
            return True
    return False


def _maybe_promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """If the first row of ``df`` looks like a header, slice it off and use it.

    The ``Unnamed: 0`` / ``Unnamed: 1`` placeholder columns that pandas
    creates when ``header=0`` is given and the first row is data are
    replaced with the actual header values. If the first row is not a
    header, the frame is returned unchanged.
    """
    if df.empty or len(df) < 2:
        return df
    first = df.iloc[0]
    if not _row_has_header_values(first):
        return df
    new_df = df.iloc[1:].copy()
    new_df.columns = [str(v).strip() if not pd.isna(v) else f"col_{i}" for i, v in enumerate(first)]
    new_df = new_df.reset_index(drop=True)
    return new_df


def _split_paired_sheet(
    df: pd.DataFrame, sheet_name: str
) -> list[tuple[str, pd.DataFrame]]:
    """If ``df`` has multiple spectra side-by-side, split into sub-frames.

    Some instruments export one worksheet containing several stacked
    peak lists (one per timepoint, labelled in the first row of each
    column group). The first column of such a sheet is a header row
    that pandas promoted; the timepoint names live in the cells of the
    first row. Each sub-frame has 6 columns, one of which is ``m/z``.

    Returns a list of ``(label, sub_df)`` tuples. If the sheet has
    only one spectrum, returns a single tuple labelled with the sheet
    name.
    """
    # Timepoint detection has to happen on the *un-promoted* frame
    # because the timepoint label is the column name that pandas
    # auto-assigned (e.g. "30m"). Once we promote the header row to
    # the column names, that label is overwritten by "m/z".
    timepoint_starts: list[tuple[int, str]] = []
    for i, col in enumerate(df.columns):
        if isinstance(col, str) and _TIMEPOINT_RE.match(col.strip()):
            timepoint_starts.append((i, col.strip()))

    if len(timepoint_starts) <= 1:
        # Single spectrum: don't split, just promote the header row
        # (if there is one) and return the whole frame.
        return [(sheet_name, _maybe_promote_header(df))]

    out: list[tuple[str, pd.DataFrame]] = []
    for k, (start, label) in enumerate(timepoint_starts):
        end = timepoint_starts[k + 1][0] if k + 1 < len(timepoint_starts) else len(df.columns)
        sub = df.iloc[:, start:end]
        sub = _maybe_promote_header(sub)
        out.append((label, sub))
    return out


def _spectrum_hash(spectrum: Spectrum) -> str:
    """Stable content hash for de-duplication.

    Hashes the (mz, intensity) pairs in *sorted* order so two spectra
    with the same content but different row order produce the same
    digest. Without sorting, a re-acquisition that emits the same
    peaks in a different order would be treated as a distinct spectrum
    and slip past dedup -- the user would see two ``210m`` tabs.
    """
    h = hashlib.sha256()
    for peak in sorted(spectrum.peaks, key=lambda p: (p.mz, p.intensity)):
        h.update(f"{peak.mz:.6f}:{peak.intensity:.6f}\n".encode("utf-8"))
    return h.hexdigest()


def _dataframe_to_spectrum(
    df: pd.DataFrame, source: str = "", mz_hi: float | None = None
) -> Spectrum:
    """Convert an in-memory DataFrame into a :class:`Spectrum`.

    ``mz_hi`` is the spectrum's acquisition upper bound (the highest
    m/z the instrument actually scanned). When provided it is stored
    on the Spectrum so the screener can use it for off-edge detection
    instead of falling back to ``max(peak.mz)`` (which is peak-
    derived and can drift from the acquisition range on a sparse
    spectrum). When ``None`` the Spectrum falls back to its peak-
    derived upper bound.
    """
    mz_col, int_col = _detect_columns(df)

    # Coerce numeric and drop rows with NaN in either column.
    mz = pd.to_numeric(df[mz_col], errors="coerce")
    inten = pd.to_numeric(df[int_col], errors="coerce")
    mask = mz.notna() & inten.notna()
    mz = mz[mask]
    inten = inten[mask]

    peaks = [
        Peak(mz=float(m), intensity=float(i))
        for m, i in zip(mz.to_list(), inten.to_list())
    ]
    return Spectrum(peaks=peaks, source=source, mz_hi=mz_hi)


def _read_dataframe(path: str | os.PathLike) -> pd.DataFrame:
    """Load a tabular file into a DataFrame, dropping the index column if present."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
    elif suffix in {".xls", ".xlsx"}:
        # openpyxl handles .xlsx; pandas falls back to xlrd for .xls
        # if it is installed. The error is informative if neither works.
        engine = "openpyxl" if suffix == ".xlsx" else None
        df = pd.read_excel(p, engine=engine)
    else:
        raise ValueError(
            f"unsupported table format {suffix!r}; expected .csv, .xls or .xlsx"
        )

    # Drop the common "Unnamed: 0" index column if pandas wrote it out.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def read_peaks_csv(
    path: str | os.PathLike, *, mz_hi: float | None = None
) -> Spectrum:
    """Read a CSV peak list and return a :class:`Spectrum`.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    df = pd.read_csv(Path(path))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return _dataframe_to_spectrum(df, source=str(path), mz_hi=mz_hi)


def read_peaks_xlsx(
    path: str | os.PathLike, *, mz_hi: float | None = None
) -> Spectrum:
    """Read an Excel (.xls / .xlsx) peak list and return a :class:`Spectrum`.

    When the file has multiple sheets, only the first sheet is read. Use
    :func:`read_xlsx_sheets` if you need them all.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    engine = "openpyxl" if suffix == ".xlsx" else None
    df = pd.read_excel(p, engine=engine)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return _dataframe_to_spectrum(df, source=str(path), mz_hi=mz_hi)


def read_xlsx_sheets(
    path: str | os.PathLike, *, mz_hi: float | None = None
) -> dict[str, Spectrum]:
    """Read every sheet of an XLSX/XLS file and return one Spectrum per sheet.

    Returns a dict ``{sheet_name: Spectrum}``. Sheets that contain no
    parseable peaks (e.g. a summary sheet without an m/z column) are
    silently skipped. Sheet-name collisions are disambiguated by
    appending `` (2)`` etc. Identical spectra (same m/z + intensity
    values) are deduplicated; the first occurrence wins.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    engine = "openpyxl" if suffix == ".xlsx" else None
    sheets = pd.read_excel(p, engine=engine, sheet_name=None)
    out: dict[str, Spectrum] = {}
    seen_hashes: set[str] = set()
    for raw_name, df in sheets.items():
        if df is None or df.empty:
            continue
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # Handle paired/side-by-side sheets (multiple timepoints per sheet)
        sub_frames = _split_paired_sheet(df, str(raw_name))
        for label, sub in sub_frames:
            try:
                spectrum = _dataframe_to_spectrum(
                    sub, source=f"{p}::{label}", mz_hi=mz_hi
                )
            except ValueError:
                continue
            if not spectrum.peaks:
                continue
            h = _spectrum_hash(spectrum)
            if h in seen_hashes:
                # Identical to one we already have -- skip duplicate
                continue
            seen_hashes.add(h)
            unique = label
            counter = 2
            while unique in out:
                unique = f"{label} ({counter})"
                counter += 1
            out[unique] = spectrum
    return out


def parse_table_sheets(
    path: str | os.PathLike, *, mz_hi: float | None = None
) -> dict[str, Spectrum]:
    """Read a tabular file and return a dict of sheet/label -> Spectrum.

    For CSV this returns a single-entry dict using the filename stem as
    the key. For Excel formats it returns one entry per sheet. This is
    the canonical multi-spectrum entry point used by the Streamlit UI.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return {
            p.stem: _dataframe_to_spectrum(
                pd.read_csv(p), source=str(p), mz_hi=mz_hi
            )
        }
    if suffix in {".xls", ".xlsx"}:
        return read_xlsx_sheets(p, mz_hi=mz_hi)
    raise ValueError(
        f"unsupported table format {suffix!r}; expected .csv, .xls or .xlsx"
    )


def parse_table(
    path: str | os.PathLike, *, mz_hi: float | None = None
) -> Spectrum:
    """Read a CSV or Excel peak list and return a :class:`Spectrum`.

    The format is inferred from the file extension. Columns are auto-
    detected by case-insensitive name; the first column is ignored when
    it looks like a pandas index (``Unnamed: 0``). Rows with NaN in
    either the m/z or intensity column are dropped.

    ``mz_hi`` is the spectrum's acquisition upper bound; see
    :meth:`Spectrum.upper_bound` for the contract. Pass ``None``
    (the default) to fall back to the peak-derived upper bound.
    """
    return _dataframe_to_spectrum(
        _read_dataframe(path), source=str(path), mz_hi=mz_hi
    )


if __name__ == "__main__":
    # Smoke test: build a tiny frame, write it to a temp CSV, parse it back.
    import tempfile

    sample = pd.DataFrame(
        {
            "m/z": [100.0, 200.0, 300.5, 405.1, 506.7],
            "Intensity": [10.0, 25.0, 12.0, 8.0, 30.0],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
        sample.to_csv(fh.name, index=False)
        tmp = fh.name

    spectrum = parse_table(tmp)
    assert len(spectrum) == len(sample), (
        f"expected {len(sample)} peaks, got {len(spectrum)}"
    )
    assert spectrum.peaks[0].mz == 100.0
    assert spectrum.peaks[-1].intensity == 30.0
    os.unlink(tmp)
    print(f"OK: parsed {len(spectrum)} peaks from {tmp!r}")
