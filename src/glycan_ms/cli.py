"""Command-line interface for the glycan mass spectrometry analyzer.

Thin wrapper over the core solver. File parsing is delegated to
``parser_table`` (CSV/XLSX/XLS) and ``parser_mzxml`` (mzXML).
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import Iterable

import click

from . import __version__
from .core import Adduct, Spectrum, solve_spectrum


_SUPPORTED_TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}
_MZXML_SUFFIXES = {".mzxml", ".mzXML"}
_MZML_SUFFIXES = {".mzml", ".mzXML"}  # .mzML also matches the xml catch-all below

# Case-insensitive adduct lookup. Keys are upper-cased.
_ADDUCT_MAP: dict[str, Adduct] = {
    "H": Adduct.H,
    "NA": Adduct.NA,
    "K": Adduct.K,
}


def _detect_format(path: Path) -> str:
    """Return one of 'csv', 'xlsx', 'xls', 'mzxml', 'mzml' based on the file suffix."""
    suffix = path.suffix.lower()
    if suffix in _SUPPORTED_TABLE_SUFFIXES:
        if suffix == ".csv":
            return "csv"
        if suffix == ".xlsx":
            return "xlsx"
        return "xls"
    if suffix in _MZXML_SUFFIXES:
        return "mzxml"
    if suffix in _MZML_SUFFIXES:
        return "mzml"
    expected = sorted(_SUPPORTED_TABLE_SUFFIXES | _MZXML_SUFFIXES | _MZML_SUFFIXES)
    raise click.BadParameter(
        f"unsupported file extension {suffix!r}; expected one of {expected}"
    )


def _parse_adducts(spec: str) -> list[Adduct]:
    """Parse a comma-separated adduct spec (e.g. 'H,Na') into a list of Adducts.

    Comparison is case-insensitive: 'H', 'h', 'H+', 'na', 'K' all work.
    """
    out: list[Adduct] = []
    for token in spec.split(","):
        key = token.strip().rstrip("+").upper()
        if not key:
            continue
        if key not in _ADDUCT_MAP:
            raise click.BadParameter(
                f"unknown adduct {token.strip()!r}; expected one of H, Na, K"
            )
        out.append(_ADDUCT_MAP[key])
    if not out:
        raise click.BadParameter("at least one adduct must be specified")
    return out


def _load_spectrum(path: Path) -> Spectrum:
    """Dispatch to the right parser based on the file suffix."""
    fmt = _detect_format(path)
    if fmt == "mzxml":
        from . import parser_mzxml
        return parser_mzxml.first_scan_spectrum(str(path))
    if fmt == "mzml":
        from . import parser_mzml
        return parser_mzml.first_scan_spectrum_mzml(str(path))
    from . import parser_table
    return parser_table.parse_table(path)


def _format_adduct(adduct: Adduct) -> str:
    """Render an Adduct enum as a short label (H/Na/K) for CSV output."""
    return adduct.name


def _write_results(out_target: str, spectrum: Spectrum, candidates: Iterable) -> None:
    """Write the candidate list as CSV.

    ``out_target`` is a path string for a file destination, or the
    literal string ``"-"`` to write to stdout.
    """
    fieldnames = [
        "mz",
        "intensity",
        "n_galnac",
        "n_gal",
        "adduct",
        "neutral_mass",
        "theoretical_mz",
        "ppm_error",
        "scan_id",
    ]
    # Build a (mz, intensity) -> scan_id lookup so duplicate (mz, intensity)
    # values still resolve to the correct scan in O(1) per candidate.
    scan_lookup: dict[tuple[float, float], int | None] = {
        (p.mz, p.intensity): p.scan_id for p in spectrum.peaks
    }

    if out_target == "-":
        fh = io.StringIO()
    else:
        fh = Path(out_target).open("w", newline="", encoding="utf-8")
    try:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            peak_scan = scan_lookup.get((cand.observed_mz, cand.intensity))
            writer.writerow(
                {
                    "mz": cand.observed_mz,
                    "intensity": cand.intensity,
                    "n_galnac": cand.n_galnac,
                    "n_gal": cand.n_gal,
                    "adduct": _format_adduct(cand.adduct),
                    "neutral_mass": cand.neutral_mass,
                    "theoretical_mz": cand.theoretical_mz,
                    "ppm_error": cand.ppm_error,
                    "scan_id": peak_scan if peak_scan is not None else "",
                }
            )
    finally:
        if out_target == "-":
            sys.stdout.write(fh.getvalue())
        else:
            fh.close()


@click.group()
@click.version_option(__version__, prog_name="glycan-ms")
def cli() -> None:
    """Glycan mass spectrometry composition analyzer."""


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--tolerance",
    default=10.0,
    show_default=True,
    type=float,
    help="Symmetric Da tolerance for composition matches.",
)
@click.option(
    "--mz-min",
    default=600.0,
    show_default=True,
    type=float,
    help="Lower m/z bound (600 is the lowest we care about).",
)
@click.option(
    "--mz-max",
    default=5000.0,
    show_default=True,
    type=float,
    help="Upper m/z bound.",
)
@click.option(
    "--adducts",
    default="H,Na,K",
    show_default=True,
    type=str,
    help="Comma-separated adducts to consider (subset of H,Na,K).",
)
@click.option(
    "--min-intensity",
    default=0.0,
    show_default=True,
    type=float,
    help="Skip peaks with intensity below this threshold.",
)
@click.option(
    "--out",
    default="results.csv",
    show_default=True,
    type=str,
    help="Output CSV path. Use - for stdout.",
)
def analyze(
    path: Path,
    tolerance: float,
    mz_min: float,
    mz_max: float,
    adducts: str,
    min_intensity: float,
    out: str,
) -> None:
    """Analyze a peak list against glycan compositions.

    PATH is a .csv, .xlsx, .xls, or .mzXML file; the format is auto-detected
    from the file suffix. Matching compositions are written to --out.
    """
    if mz_min >= mz_max:
        raise click.BadParameter("--mz-min must be < --mz-max")

    adduct_list = _parse_adducts(adducts)
    spectrum = _load_spectrum(path)
    candidates = solve_spectrum(
        spectrum,
        da_tol=tolerance,
        mz_lo=mz_min,
        mz_hi=mz_max,
        adducts=adduct_list,
        min_intensity=min_intensity,
    )
    _write_results(out, spectrum, candidates)
    if out == "-":
        click.echo(
            f"\n# Wrote {len(candidates)} candidate matches from "
            f"{len(spectrum.peaks)} peaks to stdout",
            err=True,
        )
    else:
        click.echo(
            f"Wrote {len(candidates)} candidate matches from "
            f"{len(spectrum.peaks)} peaks to {out}"
        )


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(path: Path) -> None:
    """Print summary information about a peak-list file."""
    fmt = _detect_format(path)
    spectrum = _load_spectrum(path)
    n_peaks = len(spectrum.peaks)
    if n_peaks:
        mz_values = [p.mz for p in spectrum.peaks]
        mz_lo = min(mz_values)
        mz_hi = max(mz_values)
        total_intensity = sum(p.intensity for p in spectrum.peaks)
    else:
        mz_lo = 0.0
        mz_hi = 0.0
        total_intensity = 0.0
    click.echo(f"format:       {fmt}")
    click.echo(f"peak_count:   {n_peaks}")
    click.echo(f"mz_range:     {mz_lo:.4f} - {mz_hi:.4f}")
    click.echo(f"total_intensity: {total_intensity:.4f}")


@cli.command()
def version() -> None:
    """Print the glycan-ms-analyzer package version."""
    click.echo(__version__)


def main() -> None:
    """Console-script entry point declared in pyproject.toml."""
    cli()


if __name__ == "__main__":
    main()
