// Usage: node tests/js_parity.js <model.json> <cases.json>
// cases.json: {"texts": [...], "probs": [...]} with the Python model's spam probabilities.
const fs = require("fs");
const path = require("path");
const { create } = require(path.join(__dirname, "..", "docs", "spam-model.js"));

const model = create(JSON.parse(fs.readFileSync(process.argv[2], "utf8")));
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

let maxDiff = 0, labelFlips = 0;
cases.texts.forEach((text, i) => {
  const p = model.classify(text).spam;
  maxDiff = Math.max(maxDiff, Math.abs(p - cases.probs[i]));
  if ((p >= 0.5) !== (cases.probs[i] >= 0.5)) labelFlips++;
});

console.log(JSON.stringify({ n: cases.texts.length, max_diff: maxDiff, label_flips: labelFlips }));
