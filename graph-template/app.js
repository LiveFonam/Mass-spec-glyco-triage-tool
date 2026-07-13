(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const state = {
    datasets: [], nextId: 1, pasteDatasetId: null, activeTab: null,
    palette: {}, exportGraphs: [], pasteTimer: null,
  };

  // Okabe-Ito plus other color-blind-friendly categorical colors.
  const COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000",
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255",
    "#AA4499", "#661100", "#6699CC", "#AA4466", "#4477AA", "#228833", "#EE6677", "#BBBBBB",
  ];
  const HEADER_ALIASES = {
    mz: ["mz", "m/z", "masscharge", "massovercharge", "observedmz", "measuredmz"],
    intensity: ["intensity", "signal", "abundance", "height", "peakintensity"],
    nGalNAc: ["ngalnac", "galnac", "galnaccount", "numbergalnac", "galnacunits"],
    nGal: ["ngal", "gal", "galactose", "galcount", "numbergal", "numbergalactose", "galunits"],
    total: ["total", "dp", "degreeofpolymerization", "totalsugar", "totalsugars", "totalunits", "sugarunits"],
    ion: ["ion", "adduct", "iontype", "ionform"],
  };

  function canonical(value) {
    return String(value ?? "").trim().toLowerCase().replace(/[^a-z0-9]/g, "");
  }
  const canonicalAliases = Object.fromEntries(
    Object.entries(HEADER_ALIASES).map(([field, aliases]) => [field, new Set(aliases.map(canonical))])
  );
  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
  }
  function cleanFileName(value) {
    return String(value || "graph").replace(/\.[^.]+$/, "").replace(/[^a-z0-9._-]+/gi, "_").replace(/^_+|_+$/g, "") || "graph";
  }
  function formatMz(value) {
    return Number(value).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  function formatIntensity(value) {
    return Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
  }
  function compositionKey(record) { return `${record.nGalNAc}|${record.nGal}`; }
  function galnacColorKey(recordOrCompositionKey) {
    return String(typeof recordOrCompositionKey === "string" ? recordOrCompositionKey.split("|")[0] : recordOrCompositionKey.nGalNAc);
  }
  function compositionLabel(record) { return `${record.nGalNAc} GalNAc + ${record.nGal} Gal`; }
  function sortedCompositionKeys(records) {
    return [...new Set(records.map(compositionKey))].sort((a, b) => {
      const [an, ag] = a.split("|").map(Number), [bn, bg] = b.split("|").map(Number);
      return an - bn || ag - bg;
    });
  }
  function labelFromKey(key) {
    const [nGalNAc, nGal] = key.split("|").map(Number);
    return `${nGalNAc} GalNAc + ${nGal} Gal`;
  }
  function contrastColor(hex) {
    const c = hex.replace("#", ""), r = parseInt(c.slice(0,2),16), g = parseInt(c.slice(2,4),16), b = parseInt(c.slice(4,6),16);
    return (0.299*r + 0.587*g + 0.114*b) > 155 ? "#111111" : "#ffffff";
  }
  function galnacColorMap(keys, mode = "gradient", baseColor = "#009E73", distinct = {}) {
    const ordered = [...new Set(keys.map(String))].sort((a,b)=>Number(a)-Number(b));
    if (mode === "distinct") return Object.fromEntries(ordered.map((key,index)=>[key,distinct[key]||COLORS[index%COLORS.length]]));
    const safeBase = /^#[0-9a-f]{6}$/i.test(baseColor) ? baseColor : "#009E73";
    const channels = [1,3,5].map((index)=>parseInt(safeBase.slice(index,index+2),16));
    return Object.fromEntries(ordered.map((key,index)=>{
      const concentration = ordered.length===1 ? 1 : .35 + .65*index/(ordered.length-1);
      const mixed = channels.map((channel)=>Math.round(255*(1-concentration)+channel*concentration));
      return [key,`#${mixed.map((channel)=>channel.toString(16).padStart(2,"0")).join("").toUpperCase()}`];
    }));
  }

  function setMessages(element, messages) {
    element.replaceChildren();
    for (const item of messages) {
      const div = document.createElement("div");
      div.className = `message ${item.type || ""}`.trim();
      div.textContent = item.text;
      element.append(div);
    }
  }
  function globalMessage(text, type = "") {
    setMessages($("#global-messages"), [{text, type}]);
  }

  function locateHeader(matrix) {
    for (let rowIndex = 0; rowIndex < Math.min(matrix.length, 30); rowIndex += 1) {
      const row = matrix[rowIndex] || [], mapping = {};
      row.forEach((cell, index) => {
        const key = canonical(cell);
        for (const [field, aliases] of Object.entries(canonicalAliases)) {
          if (!(field in mapping) && aliases.has(key)) mapping[field] = index;
        }
      });
      if (Object.keys(canonicalAliases).every((field) => field in mapping)) return {rowIndex, mapping};
    }
    return null;
  }

  function validateRecord(values, location, warnings) {
    const mz = Number(values.mz), intensity = Number(values.intensity);
    const nGalNAc = Number(values.nGalNAc), nGal = Number(values.nGal), total = Number(values.total);
    const ion = String(values.ion ?? "").trim();
    const reject = (reason) => { warnings.push(`${location}: ${reason}; record skipped.`); return null; };
    if (![mz, intensity, nGalNAc, nGal, total].every(Number.isFinite)) return reject("one or more numeric fields are invalid");
    if (mz <= 0 || intensity < 0) return reject("m/z must be positive and intensity cannot be negative");
    if (![nGalNAc, nGal, total].every(Number.isInteger) || nGalNAc < 0 || nGal < 0 || total < 0) return reject("GalNAc, Gal, and total must be non-negative integers");
    if (total !== nGalNAc + nGal) return reject(`total ${total} does not equal GalNAc + Gal (${nGalNAc + nGal})`);
    if (!ion) return reject("ion is empty");
    return {mz, intensity, nGalNAc, nGal, total, ion};
  }

  function parseMatrix(matrix, sourceName) {
    const warnings = [], header = locateHeader(matrix);
    if (!header) return {records: [], warnings: [`${sourceName}: required columns were not found. Expected m/z, intensity, GalNAc, Gal, total, and ion.`]};
    const records = [];
    for (let i = header.rowIndex + 1; i < matrix.length; i += 1) {
      const row = matrix[i] || [];
      if (row.every((value) => value === null || value === undefined || String(value).trim() === "")) continue;
      const values = Object.fromEntries(Object.entries(header.mapping).map(([field, index]) => [field, row[index]]));
      const record = validateRecord(values, `${sourceName} row ${i + 1}`, warnings);
      if (record) records.push(record);
    }
    return {records, warnings};
  }

  function parseCsvLine(line, delimiter) {
    const cells = []; let value = "", quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"' && quoted && line[i + 1] === '"') { value += '"'; i += 1; }
      else if (char === '"') quoted = !quoted;
      else if (char === delimiter && !quoted) { cells.push(value.trim()); value = ""; }
      else value += char;
    }
    cells.push(value.trim());
    return cells;
  }
  function detectDelimiter(lines) {
    const candidates = [",", "\t", ";"];
    return candidates.map((delimiter) => ({delimiter, count: (lines[0] || "").split(delimiter).length - 1}))
      .sort((a,b) => b.count - a.count)[0];
  }
  function parseText(text, sourceName) {
    const rawLines = String(text).replace(/^\uFEFF/, "").split(/\r?\n/);
    const nonempty = rawLines.map((line) => line.trim()).filter(Boolean);
    if (!nonempty.length) return {records: [], warnings: []};
    const detected = detectDelimiter(nonempty);
    if (detected.count > 0) return parseMatrix(nonempty.map((line) => parseCsvLine(line, detected.delimiter)), sourceName);

    const warnings = [], records = [], completeLength = Math.floor(nonempty.length / 6) * 6;
    if (nonempty.length % 6) warnings.push(`${sourceName}: ${nonempty.length % 6} trailing line(s) do not form a complete six-line record and were skipped.`);
    for (let i = 0; i < completeLength; i += 6) {
      const record = validateRecord({mz:nonempty[i], intensity:nonempty[i+1], nGalNAc:nonempty[i+2], nGal:nonempty[i+3], total:nonempty[i+4], ion:nonempty[i+5]}, `${sourceName} record ${i/6 + 1}`, warnings);
      if (record) records.push(record);
    }
    return {records, warnings};
  }

  async function parseFile(file) {
    const extension = file.name.split(".").pop().toLowerCase();
    if (["xlsx", "xls"].includes(extension)) {
      if (!window.XLSX) throw new Error("Excel support did not load. Use the offline edition or check the internet connection.");
      const workbook = XLSX.read(await file.arrayBuffer(), {type:"array"});
      if (!workbook.SheetNames.length) return {records:[], warnings:[`${file.name}: workbook has no sheets.`]};
      const sheet = workbook.Sheets[workbook.SheetNames[0]];
      const matrix = XLSX.utils.sheet_to_json(sheet, {header:1, defval:null, raw:true});
      return parseMatrix(matrix, `${file.name} / ${workbook.SheetNames[0]}`);
    }
    return parseText(await file.text(), file.name);
  }

  function addDataset(name, parsed, options = {}) {
    if (!parsed.records.length) {
      globalMessage(parsed.warnings[0] || `${name}: no valid records found.`, "error");
      return null;
    }
    const dataset = {
      id: state.nextId++, name: name.replace(/\.[^.]+$/, ""), subtitle: "", records: parsed.records,
      warnings: parsed.warnings, showPeaks: true, showProportions: true, normalize: "raw", dedup: "highest",
      peakXLabel: "m/z", peakYLabel: "Intensity", proportionXLabel: "Degree of Polymerization (DP)",
      proportionYLabel: "Proportion of total signal",
      exportSelected: {peaks: false, proportions: false},
      isPaste: Boolean(options.isPaste),
    };
    state.datasets.push(dataset);
    if (dataset.isPaste) state.pasteDatasetId = dataset.id;
    state.activeTab ??= dataset.id;
    globalMessage(`${dataset.name}: loaded ${dataset.records.length.toLocaleString()} valid record(s).`, "success");
    renderAll();
    return dataset;
  }

  async function importFiles(files) {
    for (const file of files) {
      try { addDataset(file.name, await parseFile(file)); }
      catch (error) { globalMessage(`${file.name}: ${error.message}`, "error"); }
    }
  }

  function updatePastedDataset() {
    const text = $("#paste-input").value, existing = state.datasets.find((d) => d.id === state.pasteDatasetId);
    if (!text.trim()) {
      if (existing) { state.datasets = state.datasets.filter((d) => d.id !== existing.id); state.pasteDatasetId = null; renderAll(); }
      return;
    }
    const parsed = parseText(text, "Pasted data"), name = $("#paste-name").value.trim() || "Pasted Dataset";
    if (!parsed.records.length) {
      if (existing) {
        state.datasets = state.datasets.filter((dataset) => dataset.id !== existing.id);
        state.pasteDatasetId = null;
        if (state.activeTab === existing.id) state.activeTab = state.datasets[0]?.id || null;
        renderAll();
      }
      globalMessage(parsed.warnings[0] || "Pasted data has no complete valid records.", "error");
      return;
    }
    if (existing) {
      existing.name = name; existing.records = parsed.records; existing.warnings = parsed.warnings; renderAll();
    } else addDataset(name, parsed, {isPaste:true});
  }

  function effectiveRecords(dataset) {
    const sorted = [...dataset.records].sort((a,b) => a.mz - b.mz);
    if (dataset.dedup === "all") return sorted;
    const clusters = [];
    for (const record of sorted) {
      const cluster = clusters.at(-1);
      if (!cluster || record.mz - cluster.anchor > 0.010000001) clusters.push({anchor: record.mz, records: [record]});
      else cluster.records.push(record);
    }
    return clusters.map((cluster) => {
      const largest = cluster.records.reduce((best, record) => record.intensity > best.intensity ? record : best);
      if (dataset.dedup === "highest") return {...largest};
      return {...largest, intensity: cluster.records.reduce((sum, record) => sum + record.intensity, 0)};
    });
  }

  function proportionSummary(records) {
    const grandTotal = records.reduce((sum, record) => sum + record.intensity, 0);
    const dpTotals = new Map(), compositionTotals = new Map();
    for (const record of records) {
      dpTotals.set(record.total, (dpTotals.get(record.total) || 0) + record.intensity);
      const key = compositionKey(record), byDp = compositionTotals.get(key) || new Map();
      byDp.set(record.total, (byDp.get(record.total) || 0) + record.intensity);
      compositionTotals.set(key, byDp);
    }
    return {grandTotal, dpTotals, compositionTotals};
  }

  function allEffectiveRecords() { return state.datasets.flatMap(effectiveRecords); }
  function sharedPeakRange() {
    const records = allEffectiveRecords();
    if (!records.length) return [0,1];
    if ($("#x-mode").value === "manual") {
      const lo = Number($("#x-min").value), hi = Number($("#x-max").value);
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return [lo,hi];
    }
    let lo = records[0].mz, hi = records[0].mz;
    for (const record of records) { lo = Math.min(lo, record.mz); hi = Math.max(hi, record.mz); }
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.015;
    return [lo - pad, hi + pad];
  }
  function sharedDpRange() {
    const records = state.datasets.flatMap((dataset) => dataset.records);
    if (!records.length) return [0,1];
    let lo = records[0].total, hi = records[0].total;
    for (const record of records) { lo = Math.min(lo, record.total); hi = Math.max(hi, record.total); }
    return [lo, hi];
  }

  function ensurePalette() {
    const keys = [...new Set(state.datasets.flatMap((d) => d.records).map(galnacColorKey))].sort((a,b)=>Number(a)-Number(b));
    keys.forEach((key, index) => { if (!state.palette[key]) state.palette[key] = COLORS[index % COLORS.length]; });
    return keys;
  }
  function renderPalette() {
    const keys = ensurePalette(), panel = $("#palette-panel"), editor = $("#palette-editor");
    const mode=$("#color-mode").value;panel.classList.toggle("hidden", !keys.length);editor.classList.toggle("hidden",mode==="gradient");$("#base-color").disabled=mode!=="gradient";editor.replaceChildren();
    for (const key of keys) {
      const label = document.createElement("label"); label.className = "color-field";
      const input = document.createElement("input"); input.type = "color"; input.value = state.palette[key];
      input.addEventListener("change", () => { state.palette[key] = input.value; renderAll(); });
      const text = document.createElement("span"); text.textContent = `${key} GalNAc`;
      label.append(input, text); editor.append(label);
    }
  }

  function basePlotLayout(title, subtitle) {
    const subtitleHtml = subtitle ? `<br><sup>${escapeHtml(subtitle)}</sup>` : "";
    return {
      title:{text:`${escapeHtml(title)}${subtitleHtml}`, x:.5, xanchor:"center", font:{family:"Arial, Helvetica, sans-serif", size:18, color:"#18212b"}},
      font:{family:"Arial, Helvetica, sans-serif", color:"#18212b"}, paper_bgcolor:"#ffffff", plot_bgcolor:"#ffffff",
      margin:{l:72,r:25,t:subtitle ? 85 : 68,b:70}, hovermode:"closest", dragmode:"zoom",
      xaxis:{showgrid:true, gridcolor:"rgba(130,140,150,.18)", zeroline:false, linecolor:"#7f8992", mirror:false},
      yaxis:{showgrid:true, gridcolor:"rgba(130,140,150,.22)", zeroline:false, rangemode:"tozero"},
    };
  }
  const plotConfig = {responsive:true, scrollZoom:true, displaylogo:false, doubleClick:"reset", modeBarButtonsToRemove:["select2d","lasso2d"]};

  function renderPeakPlot(div, dataset, records, xRange) {
    const normalized = dataset.normalize === "normalized", maximum = records.reduce((value, record) => Math.max(value, record.intensity), 1);
    const yValues = records.map((r) => normalized ? r.intensity / maximum * 100 : r.intensity);
    const lineX = [], lineY = [];
    records.forEach((record, i) => { lineX.push(record.mz, record.mz, null); lineY.push(0, yValues[i], null); });
    const custom = records.map((r) => [r.intensity, r.nGalNAc, r.nGal, r.total, r.ion]);
    const traces = [
      {type:"scatter", mode:"lines", x:lineX, y:lineY, line:{color:"rgba(70,76,82,.66)",width:1}, hoverinfo:"skip", showlegend:false, name:"Peaks"},
      {type:"scatter", mode:"markers+text", x:records.map((r)=>r.mz), y:yValues, text:records.map((r)=>formatMz(r.mz)), textposition:"top center",
        textfont:{size:9,color:"#4a5056"}, marker:{size:6,color:"#555b61"}, customdata:custom, cliponaxis:false, showlegend:false,
        hovertemplate:"m/z %{x:.4f}<br>intensity %{customdata[0]:,.2f}<br>GalNAc %{customdata[1]}<br>Gal %{customdata[2]}<br>total %{customdata[3]}<br>ion %{customdata[4]}<extra></extra>"},
    ];
    const layout = basePlotLayout(`${dataset.name} — Characteristic Sugar Peaks`, dataset.subtitle);
    Object.assign(layout, {height:480, showlegend:false});
    Object.assign(layout.xaxis, {title:{text:escapeHtml(dataset.peakXLabel)}, range:xRange});
    Object.assign(layout.yaxis, {title:{text:escapeHtml(dataset.peakYLabel)}, ticksuffix:normalized ? "%" : ""});
    Plotly.newPlot(div, traces, layout, plotConfig);
    div._fullExportRange = [...xRange];
    div._fullExportYRange = [0, yValues.reduce((value, y) => Math.max(value, y), 1) * 1.15];
  }

  function internalPercent(value) { return value >= 10 ? `${Math.round(value)}%` : `${value.toFixed(1)}%`; }
  function proportionTextSize(value) {
    if (value < 5) return 6;
    if (value < 12) return 8;
    if (value < 25) return 10;
    return 12;
  }
  function renderProportionPlot(div, dataset, records, dpRange) {
    const [dpMin, dpMax] = dpRange, dps = Array.from({length:dpMax-dpMin+1},(_,i)=>dpMin+i);
    const summary = proportionSummary(records), grandTotal = summary.grandTotal || 1;
    const {dpTotals, compositionTotals} = summary;
    const shownGalnacCounts = new Set(), colors = galnacColorMap(ensurePalette(),$("#color-mode").value,$("#base-color").value,state.palette);
    const traces = sortedCompositionKeys(records).map((key) => {
      const galnacKey = galnacColorKey(key), map = compositionTotals.get(key), color = colors[galnacKey], x=[],y=[],text=[],textSizes=[],custom=[];
      for (const dp of dps) {
        const intensity = map.get(dp)||0, within = dpTotals.get(dp) ? intensity/dpTotals.get(dp)*100 : 0;
        x.push(dp); y.push(intensity/grandTotal*100); text.push(intensity ? internalPercent(within) : ""); textSizes.push(proportionTextSize(within)); custom.push([within,intensity]);
      }
      const showlegend = !shownGalnacCounts.has(galnacKey); shownGalnacCounts.add(galnacKey);
      return {type:"bar",name:`${galnacKey} GalNAc`,legendgroup:`galnac-${galnacKey}`,showlegend,x,y,text,textposition:"inside",insidetextanchor:"middle",textangle:0,textfont:{color:contrastColor(color),size:textSizes},
        marker:{color,line:{color:"#ffffff",width:1}},customdata:custom,
        hovertemplate:"DP %{x}<br>"+escapeHtml(labelFromKey(key))+"<br>%{y:.2f}% of total signal<br>%{customdata[0]:.2f}% within DP<br>summed intensity %{customdata[1]:,.2f}<extra></extra>"};
    });
    const annotations = dps.filter((dp)=>dpTotals.get(dp)>0).map((dp)=>({x:dp,y:dpTotals.get(dp)/grandTotal*100,text:`<b>${(dpTotals.get(dp)/grandTotal*100).toFixed(1)}%</b>`,showarrow:false,yshift:10,font:{size:11,color:"#111"}}));
    const maxShare = Math.max(...dps.map((dp)=>(dpTotals.get(dp)||0)/grandTotal*100),1);
    const layout = basePlotLayout(`${dataset.name} — Composition Proportions`, dataset.subtitle);
    Object.assign(layout,{height:510,barmode:"stack",bargap:.12,annotations,showlegend:true,
      uniformtext:{mode:"show",minsize:6},
      legend:{orientation:"h",x:.5,xanchor:"center",y:-.23,yanchor:"top",title:{text:"GalNAc count"}},
      margin:{l:78,r:25,t:dataset.subtitle?85:68,b:125}});
    Object.assign(layout.xaxis,{title:{text:escapeHtml(dataset.proportionXLabel)},range:[dpMin-.5,dpMax+.5],tickmode:"linear",dtick:1});
    Object.assign(layout.yaxis,{title:{text:escapeHtml(dataset.proportionYLabel)},ticksuffix:"%",range:[0,maxShare*1.18]});
    Plotly.newPlot(div,traces,layout,plotConfig); div._fullExportRange=[dpMin-.5,dpMax+.5]; div._fullExportYRange=[0,maxShare*1.18];
  }

  function fillTable(body, records) {
    body.replaceChildren();
    const limit = Math.min(records.length,5000);
    for (let i=0;i<limit;i+=1) {
      const r=records[i], row=document.createElement("tr");
      [formatMz(r.mz),formatIntensity(r.intensity),r.nGalNAc,r.nGal,r.total,r.ion].forEach((value)=>{const td=document.createElement("td");td.textContent=value;row.append(td);});
      body.append(row);
    }
    if (records.length > limit) {
      const row=document.createElement("tr"),cell=document.createElement("td");cell.colSpan=6;
      cell.textContent=`Showing the first ${limit.toLocaleString()} of ${records.length.toLocaleString()} records.`;row.append(cell);body.append(row);
    }
  }

  function attachDatasetEvents(card, dataset) {
    const rerender = () => renderAll();
    const title=$(".dataset-title",card), subtitle=$(".dataset-subtitle",card);
    title.addEventListener("change",()=>{dataset.name=title.value.trim()||"Untitled Dataset";rerender();});
    subtitle.addEventListener("change",()=>{dataset.subtitle=subtitle.value.trim();rerender();});
    $(".show-peaks",card).addEventListener("change",(e)=>{dataset.showPeaks=e.target.checked;rerender();});
    $(".show-proportions",card).addEventListener("change",(e)=>{dataset.showProportions=e.target.checked;rerender();});
    $(".normalization-mode",card).addEventListener("change",(e)=>{dataset.normalize=e.target.value;rerender();});
    $(".dedup-mode",card).addEventListener("change",(e)=>{dataset.dedup=e.target.value;rerender();});
    for (const [selector, field] of [[".peak-x-label","peakXLabel"],[".peak-y-label","peakYLabel"],[".proportion-x-label","proportionXLabel"],[".proportion-y-label","proportionYLabel"]]) {
      $(selector,card).addEventListener("change",(event)=>{dataset[field]=event.target.value.trim();rerender();});
    }
    $(".remove-dataset",card).addEventListener("click",()=>{state.datasets=state.datasets.filter((d)=>d.id!==dataset.id);if(state.pasteDatasetId===dataset.id){state.pasteDatasetId=null;$("#paste-input").value="";}if(state.activeTab===dataset.id)state.activeTab=state.datasets[0]?.id||null;rerender();});
  }

  function renderTabs() {
    const nav=$("#dataset-tabs"), mode=$("#layout-mode").value; nav.replaceChildren(); nav.classList.toggle("hidden",mode!=="tabs"||!state.datasets.length);
    if(mode!=="tabs")return;
    if(!state.datasets.some((d)=>d.id===state.activeTab))state.activeTab=state.datasets[0]?.id;
    for(const dataset of state.datasets){const b=document.createElement("button");b.className=`tab-button ${dataset.id===state.activeTab?"active":""}`;b.textContent=dataset.name;b.addEventListener("click",()=>{state.activeTab=dataset.id;renderAll();});nav.append(b);}
  }

  function renderAll() {
    if(!window.Plotly){globalMessage("Plotly did not load. Open the offline edition or check the internet connection.","error");return;}
    renderPalette(); state.exportGraphs=[];
    const container=$("#datasets"), mode=$("#layout-mode").value;
    container.className=`datasets ${mode}-layout`; container.replaceChildren(); renderTabs();
    if(!state.datasets.length){const empty=document.createElement("div");empty.id="empty-state";empty.className="empty-state";empty.innerHTML='<div class="empty-symbol">⌁</div><h2>No datasets loaded</h2><p>Upload a file or paste records above to generate graphs.</p>';container.append(empty);updateSelectionCount();return;}
    const xRange=sharedPeakRange(),dpRange=sharedDpRange(),template=$("#dataset-template");
    for(const dataset of state.datasets){
      const card=template.content.firstElementChild.cloneNode(true);card.dataset.id=dataset.id;
      if(mode==="tabs"&&dataset.id!==state.activeTab)card.classList.add("tab-hidden");
      $(".dataset-title",card).value=dataset.name;$(".dataset-subtitle",card).value=dataset.subtitle;
      $(".show-peaks",card).checked=dataset.showPeaks;$(".show-proportions",card).checked=dataset.showProportions;
      $(".normalization-mode",card).value=dataset.normalize;$(".dedup-mode",card).value=dataset.dedup;
      $(".peak-x-label",card).value=dataset.peakXLabel;$(".peak-y-label",card).value=dataset.peakYLabel;
      $(".proportion-x-label",card).value=dataset.proportionXLabel;$(".proportion-y-label",card).value=dataset.proportionYLabel;
      const records=effectiveRecords(dataset),combinedIntensity=dataset.records.reduce((s,r)=>s+r.intensity,0);
      $(".dataset-summary",card).textContent=`${records.length.toLocaleString()} displayed peak(s) · ${dataset.records.length.toLocaleString()} imported record(s) · total intensity ${formatIntensity(combinedIntensity)}`;
      setMessages($(".dataset-warnings",card),dataset.warnings.map((text)=>({text})));
      const peakSection=$(".peaks-section",card),propSection=$(".proportions-section",card);
      peakSection.classList.toggle("hidden",!dataset.showPeaks);propSection.classList.toggle("hidden",!dataset.showProportions);
      fillTable($(".data-table-body",card),records);attachDatasetEvents(card,dataset);container.append(card);
      if(dataset.showPeaks){const div=$(".peak-plot",card);renderPeakPlot(div,dataset,records,xRange);registerExport(dataset,"peaks",div,$(".select-peak-export",card),$(".export-peak",card));}
      if(dataset.showProportions){const div=$(".proportion-plot",card);renderProportionPlot(div,dataset,dataset.records,dpRange);registerExport(dataset,"proportions",div,$(".select-proportion-export",card),$(".export-proportion",card));}
    }
    updateSelectionCount();
  }

  function registerExport(dataset,type,div,checkbox,button){
    const graph={dataset,type,div,checkbox,fileName:()=>`${cleanFileName(dataset.name)}_${type}.png`};state.exportGraphs.push(graph);
    checkbox.checked=dataset.exportSelected[type];
    checkbox.addEventListener("change",()=>{dataset.exportSelected[type]=checkbox.checked;updateSelectionCount();});button.addEventListener("click",()=>downloadGraph(graph));
  }
  function updateSelectionCount(){const count=state.exportGraphs.filter((g)=>g.checkbox.checked).length;$("#selection-count").textContent=`${count} graph${count===1?"":"s"} selected`;}
  function exportDimensions(){return {width:Math.max(400,Math.min(10000,Number($("#png-width").value)||1920)),height:Math.max(300,Math.min(10000,Number($("#png-height").value)||1080))};}
  async function graphDataUrl(graph){
    const div=graph.div,currentX=div.layout?.xaxis?.range?[...div.layout.xaxis.range]:null,currentY=div.layout?.yaxis?.range?[...div.layout.yaxis.range]:null;
    const restore={};if(currentX)restore["xaxis.range"]=currentX;else restore["xaxis.autorange"]=true;if(currentY)restore["yaxis.range"]=currentY;else restore["yaxis.autorange"]=true;
    try {
      await Plotly.relayout(div,{"xaxis.range":div._fullExportRange,"yaxis.range":div._fullExportYRange,"paper_bgcolor":"#ffffff","plot_bgcolor":"#ffffff"});
      return await Plotly.toImage(div,{format:"png",...exportDimensions()});
    } finally {
      await Plotly.relayout(div,restore);
    }
  }
  function triggerDownload(url,fileName){const a=document.createElement("a");a.href=url;a.download=fileName;document.body.append(a);a.click();a.remove();}
  async function downloadGraph(graph){try{triggerDownload(await graphDataUrl(graph),graph.fileName());}catch(error){globalMessage(`PNG export failed: ${error.message}`,"error");}}
  function selectedGraphs(){return state.exportGraphs.filter((g)=>g.checkbox.checked);}
  async function downloadSelectedSeparately(){const graphs=selectedGraphs();if(!graphs.length){globalMessage("Select at least one graph for export.","error");return;}for(const graph of graphs){await downloadGraph(graph);await new Promise((resolve)=>setTimeout(resolve,250));}}
  async function downloadSelectedZip(){
    const graphs=selectedGraphs();if(!graphs.length){globalMessage("Select at least one graph for export.","error");return;}if(!window.JSZip){globalMessage("ZIP support did not load. Use separate downloads or the offline edition.","error");return;}
    try{const zip=new JSZip(),usedNames=new Map();for(const graph of graphs){const url=await graphDataUrl(graph),original=graph.fileName(),count=(usedNames.get(original)||0)+1;usedNames.set(original,count);const name=count===1?original:original.replace(/\.png$/i,`_${count}.png`);zip.file(name,url.split(",")[1],{base64:true});}const blob=await zip.generateAsync({type:"blob",compression:"DEFLATE"});triggerDownload(URL.createObjectURL(blob),"glycan_graphs.zip");globalMessage(`Exported ${graphs.length} PNG(s) in glycan_graphs.zip.`,"success");}catch(error){globalMessage(`ZIP export failed: ${error.message}`,"error");}
  }

  function bindEvents(){
    const input=$("#file-input"),drop=$("#file-drop");input.addEventListener("change",()=>importFiles(input.files));
    ["dragenter","dragover"].forEach((name)=>drop.addEventListener(name,(event)=>{event.preventDefault();drop.classList.add("dragging");}));
    ["dragleave","drop"].forEach((name)=>drop.addEventListener(name,(event)=>{event.preventDefault();drop.classList.remove("dragging");}));
    drop.addEventListener("drop",(event)=>importFiles(event.dataTransfer.files));
    $("#paste-input").addEventListener("input",()=>{clearTimeout(state.pasteTimer);state.pasteTimer=setTimeout(updatePastedDataset,300);});
    $("#paste-name").addEventListener("input",()=>{const d=state.datasets.find((item)=>item.id===state.pasteDatasetId);if(d){d.name=$("#paste-name").value.trim()||"Pasted Dataset";clearTimeout(state.pasteTimer);state.pasteTimer=setTimeout(renderAll,300);}});
    $("#layout-mode").addEventListener("change",renderAll);$("#x-mode").addEventListener("change",(event)=>{$("#x-min").disabled=event.target.value!=="manual";$("#x-max").disabled=event.target.value!=="manual";renderAll();});
    ["#x-min","#x-max"].forEach((selector)=>$(selector).addEventListener("change",renderAll));
    $("#color-mode").addEventListener("change",renderAll);$("#base-color").addEventListener("input",renderAll);
    $("#reset-colors").addEventListener("click",()=>{state.palette={};$("#color-mode").value="gradient";$("#base-color").value="#009E73";renderAll();});
    $("#export-separate").addEventListener("click",downloadSelectedSeparately);$("#export-zip").addEventListener("click",downloadSelectedZip);
    window.addEventListener("resize",()=>state.exportGraphs.forEach((graph)=>Plotly.Plots.resize(graph.div)));
  }

  globalThis.GlycanGraphTemplate = {parseText, parseMatrix, validateRecord, effectiveRecords, proportionSummary, galnacColorKey, galnacColorMap, proportionTextSize};
  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded",()=>{bindEvents();renderAll();});
  }
})();
