# glyco-grade

`glyco-grade` is a small, focused Python library and Streamlit app for
**scoring GalNAc/Gal O-glycan composition candidates** from MS peak
lists (CSV, Excel, or mzXML). The library does not call peaks, identify
unknowns, or do structural elucidation: it takes candidate compositions
the upstream solver already matched and grades them against a small set
of biological signals (windowed noise floor, 13C isotope envelope,
companion peaks, and series / 3-in-a-row ladders) and writes a tiered
table the user can read.

The output is a **GREEN / YELLOW / RED / PURGE** tier per candidate
plus a human-readable note explaining the score. The tool is intended
as a no-cost, dependency-minimal complement to heavier tools such as
GlycoWorkbench, designed for quick triage of O-glycan profiling data
on a laptop.

## Install

```bash
python -m pip install -e .
```

The package pulls in `numpy`, `pandas`, `openpyxl` and `click` at install
time. For the test suite, also install the optional `test` extra:

```bash
python -m pip install -e ".[test]"
```

## Quick start

Analyze a peak list straight from the command line:

```bash
glycan-ms analyze data/peaks.csv --tolerance 0.7 --min-intensity 100 --out results.csv
```

Or use the library directly:

```python
from glycan_ms import (
    Adduct, Peak, Spectrum, solve_peak, solve_spectrum, ppm_error,
)

# Single peak
cands = solve_peak(
    mz=1177.4236,
    intensity=1000.0,
    da_tol=0.7,
    mz_lo=1100.0,
    mz_hi=1300.0,
    adducts=[Adduct.NA],
)
for c in cands:
    print(c.n_galnac, c.n_gal, c.adduct, c.ppm_error)

# Whole spectrum
spectrum = Spectrum(peaks=[Peak(mz=1177.4236, intensity=1000.0)])
matches = solve_spectrum(spectrum, da_tol=0.7, min_intensity=0.0)
```

## Input formats

| Format | File extension | Notes                                                                                  |
|--------|----------------|----------------------------------------------------------------------------------------|
| CSV    | `.csv`         | Two columns: `m/z` and `intensity`. Column names are matched case-insensitively.       |
| Excel  | `.xlsx`        | Same layout as CSV. Uses `openpyxl` under the hood.                                    |
| Excel  | `.xls`         | Same layout as CSV. Requires `xlrd` for legacy files (not installed by default).        |
| mzXML  | `.mzXML`       | Peak list read from the first MS1 scan. Full scan-by-scan parsing is on the roadmap.   |

## Chemistry

The composition is solved against the following monoisotopic masses
(ExPASy / GlycoMod standard values).

| Building block              | Monoisotopic mass (Da) |
|-----------------------------|------------------------|
| Galactose / Hex residue     | 162.0528               |
| GalNAc / HexNAc residue     | 203.0794               |
| H2O (free reducing end)     | 18.0106                |
| Proton adduct `[M+H]+`      | 1.0073                 |
| Sodium adduct `[M+Na]+`     | 22.9898                |
| Potassium adduct `[M+K]+`   | 38.9637                |

The neutral mass is

```
M_neutral = n * 203.0794 + m * 162.0528 + 18.0106
```

and the observed m/z is `M_neutral + adduct.mass`. The solver reports
the signed ppm error of each candidate as

```
ppm_error = (observed - theoretical) / theoretical * 1e6
```

Positive values mean the measurement is heavier than the proposed
composition; negative values mean it is lighter.

## Analysis profiles and graph comparison

The Streamlit sidebar includes an optional **GalNAc-biased interpretation**
profile. It keeps exact-mass candidates only when the GalNAc-labelled HexNAc
count is at least one and is greater than or equal to the Gal count. This is a
biological interpretation filter for GalNAc-rich O-glycan samples. It does not
claim to distinguish GalNAc from GlcNAc from MS1 mass because both contribute
the same HexNAc residue mass in this solver.

For comparisons across datasets, **Use the same X-axis ranges for every
sample** applies a user-defined shared m/z range to the raw spectrum and
characteristic-peak graphs, plus a user-defined shared DP range to composition
proportion graphs. The **Compare samples** control accepts any number of
additional uploaded samples and renders each with the same shared ranges.

## CLI reference

The installed console script is `glycan-ms`. It exposes two subcommands.

### `glycan-ms analyze`

Match peaks from a peak-list file to candidate compositions and write a
CSV of results.

```bash
glycan-ms analyze INPUT [OPTIONS]
```

| Flag             | Description                                                              | Default    |
|------------------|--------------------------------------------------------------------------|------------|
| `INPUT`          | Path to a CSV, XLSX or mzXML peak list (positional, required).           | -          |
| `--tolerance`    | Symmetric Da tolerance for matching.                                    | `10.0`     |
| `--mz-min`       | Lower bound of the m/z search window.                                    | `600.0`    |
| `--mz-max`       | Upper bound of the m/z search window.                                    | `5000.0`   |
| `--min-intensity`| Skip peaks with intensity below this value.                              | `0.0`      |
| `--adducts`      | Comma-separated adducts to consider. Case-insensitive: `H`, `Na`, `K`.   | `H,Na,K`   |
| `--out`, `-o`    | Output CSV path. Use `-` for stdout.                                     | `results.csv` |

### `glycan-ms info`

Print a summary of a peak-list file: format, peak count, m/z range,
and total intensity.

```bash
glycan-ms info data/peaks.csv
```

## Tests

The test suite lives in `tests/` and is run with `pytest`:

```bash
python -m pytest
```

The suite covers the signed `ppm_error` helper, the `solve_peak` algorithm
(known compositions, Da window enforcement, m/z window enforcement), the
`solve_spectrum` intensity filter, a full CSV round-trip through
`glycan_ms.parser_table.parse_table`, and an end-to-end mzXML pipeline
smoke test against `data/2c5-120m_spectrum.mzXML`.

## License

No license file is shipped at this time. Until one is added, treat the
source as "all rights reserved" by default.