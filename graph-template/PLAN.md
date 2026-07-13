# Glycan graph template implementation plan

## Objective

Build a reusable browser-based HTML template that accepts characteristic glycan peak data and produces publication-ready peak and composition-proportion graphs. Provide both a small CDN edition and a fully self-contained offline edition. Keep all input data in the browser.

## Locked design decisions

### Input data

- Required fields: `m/z`, `intensity`, `GalNAc`, `Gal`, `total`, `ion`.
- Vertical pasted/TXT records contain six lines in that exact order.
- Accept pasted text, TXT, CSV/TSV, XLS, and XLSX.
- CSV/Excel parsing recognizes common header variations.
- Excel uses the first worksheet only.
- Ignore every spreadsheet column except the six required fields.
- Reject malformed records and show a clear warning identifying the row/record.
- Reject records where `total != GalNAc + Gal`.
- Multiple records in the same 0.01 m/z bin are user-configurable: keep highest (default), sum, or keep all.
- File basename becomes the default dataset title and remains editable.
- Pasted input has an editable dataset-name field.
- Optional editable subtitle per dataset.
- Input updates automatically.

### Upper graph: Characteristic Sugar Peaks

- Fixed graph name: **Characteristic Sugar Peaks**.
- X-axis: m/z; Y-axis: Intensity.
- Thin dark-gray vertical sticks with a dark-gray dot at the top.
- Print m/z above every peak.
- Hover shows all six fields.
- Per-dataset switch between raw intensity and normalization where the maximum is 100%.
- All compared peak graphs share the same x-axis range.
- Shared range is automatic by default with optional manual minimum/maximum controls.
- Left-drag zoom, scroll-wheel zoom, toolbar pan, and double-click reset.

### Lower graph: Composition Proportions

- Fixed graph name: **Composition Proportions**.
- X-axis is total sugar length / Degree of Polymerization: `GalNAc + Gal`.
- Y-axis is the proportion of total summed intensity.
- For each DP, sum every peak intensity at that total and divide by total intensity across the dataset.
- Thick stacked bars.
- Each stacked segment represents one exact `(GalNAc, Gal)` composition, while
  its color encodes only the GalNAc count.
- If the same composition appears with multiple ions, combine their intensities.
- Every composition with the same GalNAc count uses the same color across datasets.
- Default color-blind-friendly categorical palette; every GalNAc-count color is user-editable.
- Stack/legend order: lowest GalNAc count to highest.
- Legend appears below the graph and identifies GalNAc counts.
- Show every integer DP tick between the global minimum and maximum, including empty totals.
- Top of each DP bar shows its percentage of total signal with one decimal.
- Every internal segment shows percentage within that DP; use whole numbers for values >=10%, one decimal below 10%.
- Show all internal percentages even when sections are small.
- Default editable axis labels: X **Degree of Polymerization (DP)**; Y **Proportion of total signal**.

### Multiple datasets and visibility

- Every dataset has its own peak/proportion pair, arranged top/bottom.
- Each dataset has separate **Show peaks** and **Show proportions** switches.
- Layout options: two-column grid (default), stacked, or tabs.
- Peak x-ranges are shared; DP ranges are shared.
- Y normalization remains per dataset.
- Include a collapsible parsed-data table, hidden by default.
- Default visual theme is white scientific/publication style using Arial/Helvetica.

### PNG and ZIP export

- Export individual graphs, never a forced combined peak/proportion image.
- Individual PNG button on every graph.
- Select multiple graphs and either trigger separate PNG downloads or download one ZIP.
- Default PNG resolution: 1920 × 1080; width and height are user-editable.
- White PNG background.
- Export the complete data range, not the current zoom.
- PNG includes title, subtitle, axes, labels, stacked percentages, and legend.

### Distribution

- Place all files under repository folder `graph-template/`.
- `index.html`: CDN edition.
- `index-offline.html`: fully self-contained offline edition.
- Pinned dependencies:
  - Plotly.js 3.6.0
  - SheetJS 0.20.3
  - JSZip 3.10.1
- Include README instructions and an offline build script.

## Analyser-side addition

- Noise-floor multiplier is currently tuned to `1.64` locally.
- Noise-floor interpolation/smoothing and left-drag zoom are local changes.
- New isotope rule: for candidate `m/z < 900`, M+3 must never be required, regardless of S/N or GalNAc dominance.
- A regression test for the sub-900 rule has been added; screener suite passed 146 tests after this change.

## Integrated analyser workflow

The Streamlit mass-spec analyser is now the primary graph-making workflow; the
standalone HTML template remains available for portable/offline use.

- Every candidate receives a stable internal row ID.
- The candidate table has a one-click `✕` checkbox for removing a row.
- Removal is non-destructive: **Undo last removal** and **Restore all** are
  available, while the original uploaded spectrum remains unchanged.
- Current retained/filtered rows drive:
  - candidate markers on the diagnostic raw spectrum;
  - a clean **Characteristic Sugar Peaks** graph made only from table rows;
  - the stacked **Composition Proportions** graph;
  - candidate counts and error metrics;
  - XLSX and PNG downloads.
- The curated peak graph keeps only the highest-intensity candidate within each
  0.01 m/z line, prints m/z above every peak, exposes all composition details on
  hover, supports zoom, and can normalize its maximum to 100%.
- The proportions graph combines ions for identical `(GalNAc, Gal)`
  compositions and recalculates immediately after row curation.
- Integrated regression coverage lives in
  `tests/test_composition_proportions.py`.
- Each of the three analyser graphs and the candidate table has an independent,
  persistent Show/Hide control. Hidden sections stay hidden through preparation
  and download reruns.
- The Downloads section lets the user select any subset of graphs and prepares
  only those PNGs on demand. Prepared files are added to a session-wide export
  collection that persists while the user switches between and edits datasets.
- One final ZIP contains every collected graph from every prepared dataset.
  Individual PNG downloads remain available. Filenames use the edited dataset
  name plus `_spec.png`, `_charpeaks.png`, or `_prograph.png`.
- The collection can remove individual prepared files or be cleared entirely;
  preparing the same dataset/graph type again updates that collection entry.

## Files completed so far

- `graph-template/index.html`: CDN interface structure and controls.
- `graph-template/styles.css`: responsive scientific layout.
- `graph-template/app.js`: parsing, validation, dataset state, deduplication, both Plotly graphs, palette editor, layouts, tables, PNG export, and ZIP export.
- `graph-template/README.md`: initial user documentation.
- `graph-template/PLAN.md`: this durable implementation record.

## Validation completed

1. `node --check graph-template/app.js` passes.
2. `node graph-template/test_app.js` passes coverage for vertical input, common
   CSV headers, ignored columns, invalid totals, every duplicate mode, the exact
   0.01 boundary, and intensity-weighted DP totals.
3. The three pinned browser libraries are vendored and SHA-256 verified by the
   build script.
4. `index-offline.html` is generated, about 5.7 MiB, and has no external script
   or stylesheet references.
5. Independent code audit findings were incorporated: proportions use every
   raw record, export choices persist across rerenders/tabs, ZIP names cannot
   overwrite, stale pasted graphs clear on invalid input, and PNG layout is
   restored even after an export failure.
6. Maintained Python suite: 202 passed, 1 deliberately deselected known
   sandbox-dependent round-trip test.
7. Screener suite: 146 passed, including the below-m/z-900 M+3 regression.

## Remaining manual check

- Interactively exercise graph hover, zoom, XLS/XLSX selection, and actual
  browser PNG/ZIP downloads. An isolated Firefox render verified the complete
  offline page shell and styling; source, parser, offline-asset, and Python
  checks all pass. Browser-driven graph interactions and downloads still need
  a normal interactive click-through.
- Run `leak-check` before any requested commit/push.

## Publication status

- The previous performance/graph commit was pushed as `feeda07`.
- All changes listed in this plan are currently local and uncommitted.
- Do not push until the user requests it and leak-check passes.
