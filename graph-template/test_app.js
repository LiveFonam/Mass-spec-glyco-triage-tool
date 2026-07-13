"use strict";

const assert = require("node:assert/strict");
require("./app.js");

const {parseText, parseMatrix, effectiveRecords, proportionSummary, galnacColorKey, galnacColorMap, proportionTextSize} = globalThis.GlycanGraphTemplate;

const vertical = parseText(
  "892.2627563\n16213\n1\n4\n5\nNa+\n1095.312378\n4601\n2\n4\n6\nNa+\n",
  "vertical.txt",
);
assert.equal(vertical.records.length, 2);
assert.deepEqual(vertical.records[0], {
  mz: 892.2627563, intensity: 16213, nGalNAc: 1, nGal: 4, total: 5, ion: "Na+",
});

const csv = parseText(
  "notes,m/z,peak intensity,GalNAc,galactose,DP,adduct\nkeep,900.1,50,2,3,5,H+\n",
  "headers.csv",
);
assert.equal(csv.records.length, 1);
assert.equal(csv.records[0].ion, "H+");

const invalid = parseMatrix([
  ["m/z", "intensity", "GalNAc", "Gal", "total", "ion", "ignored"],
  [800, 100, 2, 2, 5, "Na+", "extra"],
], "invalid.xlsx");
assert.equal(invalid.records.length, 0);
assert.match(invalid.warnings[0], /does not equal/);

const duplicateRecords = [
  {mz: 100.000, intensity: 10, nGalNAc: 1, nGal: 1, total: 2, ion: "H+"},
  {mz: 100.004, intensity: 25, nGalNAc: 2, nGal: 1, total: 3, ion: "Na+"},
  {mz: 101.000, intensity: 5, nGalNAc: 1, nGal: 2, total: 3, ion: "H+"},
];
const highest = effectiveRecords({records: duplicateRecords, dedup: "highest"});
assert.equal(highest.length, 2);
assert.equal(highest[0].intensity, 25);
assert.equal(highest[0].ion, "Na+");

const summed = effectiveRecords({records: duplicateRecords, dedup: "sum"});
assert.equal(summed.length, 2);
assert.equal(summed[0].intensity, 35);
assert.equal(summed[0].ion, "Na+");

const all = effectiveRecords({records: duplicateRecords, dedup: "all"});
assert.equal(all.length, 3);

const exactBoundary = effectiveRecords({records: [duplicateRecords[0], {...duplicateRecords[1], mz: 100.01}], dedup: "highest"});
assert.equal(exactBoundary.length, 1);

const proportions = proportionSummary(vertical.records);
assert.equal(proportions.grandTotal, 20814);
assert.equal(proportions.dpTotals.get(5), 16213);
assert.equal(proportions.dpTotals.get(6), 4601);
assert.equal(proportions.compositionTotals.get("1|4").get(5), 16213);
assert.equal(galnacColorKey(vertical.records[0]), "1");
assert.equal(galnacColorKey("1|4"), galnacColorKey("1|5"));
assert.notEqual(galnacColorKey("1|4"), galnacColorKey("2|4"));
const greenGradient = galnacColorMap([1,2,3]);
assert.equal(greenGradient[3], "#009E73");
assert.notEqual(greenGradient[1], greenGradient[2]);
assert.deepEqual(galnacColorMap([1,2], "distinct", "#009E73", {1:"#112233",2:"#445566"}), {1:"#112233",2:"#445566"});
assert.ok(proportionTextSize(2) < proportionTextSize(8));
assert.ok(proportionTextSize(8) < proportionTextSize(20));
assert.ok(proportionTextSize(20) < proportionTextSize(50));

console.log("graph-template parser and duplicate-handling tests passed");
