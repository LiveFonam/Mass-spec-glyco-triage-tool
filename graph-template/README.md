# Glycan graph template

This browser template turns characteristic glycan peak records into two coordinated figures per dataset:

- **Characteristic Sugar Peaks** — thin gray sticks with a dot and m/z label at every peak.
- **Composition Proportions** — stacked bars by degree of polymerization (GalNAc + Gal), weighted by summed peak intensity.

## Open the template

- `index.html` is the smaller CDN edition and needs internet access when first opened.
- `index-offline.html` is generated as a self-contained file and works without internet access.

Double-click either file to open it in Firefox. Data stays in the browser and is not uploaded by the template.

## Accepted inputs

- Pasted vertical records: m/z, intensity, GalNAc, Gal, total, ion (six lines per record).
- TXT with the same vertical layout or delimited rows.
- CSV/TSV with commonly named columns.
- XLS/XLSX; the first sheet is used.

Extra spreadsheet columns are ignored. A record is rejected when `total != GalNAc + Gal`.

The duplicate-m/z setting affects the peak display and parsed table only. The
composition-proportion graph always sums every valid imported intensity, as
required for signal proportions.

Dataset titles, subtitles, all four axis labels, peak normalization, visible
graphs, duplicate display mode, and GalNAc-count colors are editable in the
page. Exact composition remains available in the proportion-segment hover;
segments with the same GalNAc count intentionally share one color.

The default proportion style is a single green concentration scale: lower
GalNAc counts are lighter and higher counts become increasingly strong green.
The base color is editable. Switch to **Distinct colors by GalNAc count** to
choose a separate editable color for every GalNAc amount.

## Exports

Each graph can be exported separately as a white-background PNG. Select multiple graphs to download them separately or package them into one ZIP. PNG dimensions default to 1920 × 1080 and are editable.

## Rebuild the offline edition

The offline file embeds the pinned libraries in `vendor/` plus `styles.css` and `app.js`:

```powershell
python build_offline.py
```

Run that command from the `graph-template` folder (or run
`python graph-template/build_offline.py` from the repository root). The builder
verifies the vendored-library SHA-256 checksums before writing the generated
`index-offline.html`; edit the source files rather than the generated file.

Pinned browser libraries: Plotly.js 3.6.0, SheetJS 0.20.3, and JSZip 3.10.1.
The generated offline edition is about 5.7 MiB and contains no external script
or stylesheet references.

## Developer checks

```powershell
node --check app.js
node test_app.js
python build_offline.py
```
