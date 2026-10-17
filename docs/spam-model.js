// JavaScript port of the scikit-learn pipeline in this repo: TF-IDF features (words, word pairs and
// character n-grams) followed by a linear classifier. It reads docs/model.json, which is written by
// export_web_model.py. Works in the browser (window.SpamModel) and in Node (require), where the
// tests use it to check that it gives the same probabilities as the Python model.
(function (root) {
  "use strict";

  // Same as sklearn's token_pattern r"(?u)\b\w\w+\b": runs of 2+ word characters.
  const WORD = /[\p{L}\p{N}_]{2,}/gu;

  function create(model) {
    const stopWords = new Set(model.stop_words);
    const wordIndex = new Map(model.word.terms.map((t, i) => [t, i]));
    const charIndex = model.char ? new Map(model.char.terms.map((t, i) => [t, i])) : null;
    const [wordMin, wordMax] = model.word.ngram_range;

    // sklearn analyzer="word": lowercase, tokenize, drop stop words, then build n-grams.
    function wordFeatures(text) {
      const tokens = (text.toLowerCase().match(WORD) || []).filter(w => !stopWords.has(w));
      const out = [];
      for (let n = wordMin; n <= wordMax; n++)
        for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
      return out;
    }

    // sklearn analyzer="char_wb": n-grams inside each word padded with spaces.
    function charFeatures(text) {
      const [min, max] = model.char.ngram_range;
      const out = [];
      for (const word of text.toLowerCase().replace(/\s\s+/g, " ").split(/\s+/).filter(Boolean)) {
        const w = Array.from(" " + word + " ");
        for (let n = min; n <= max; n++) {
          let offset = 0;
          out.push(w.slice(0, n).join(""));
          while (offset + n < w.length) { offset++; out.push(w.slice(offset, offset + n).join("")); }
          if (offset === 0) break; // a short word is counted only once
        }
      }
      return out;
    }

    // Sublinear TF * IDF, then L2-normalised. Returns [featureIndex, value] pairs.
    function tfidf(features, index, idf) {
      const counts = new Map();
      for (const f of features) {
        const i = index.get(f);
        if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
      }
      let norm = 0;
      const pairs = [];
      for (const [i, tf] of counts) {
        const v = (1 + Math.log(tf)) * idf[i];
        pairs.push([i, v]);
        norm += v * v;
      }
      norm = Math.sqrt(norm);
      return norm === 0 ? [] : pairs.map(([i, v]) => [i, v / norm]);
    }

    // Returns the spam probability and how much each known word (or word pair) pushed it.
    function classify(text) {
      let score = model.intercept;
      const contributions = [];
      for (const [i, v] of tfidf(wordFeatures(text), wordIndex, model.word.idf)) {
        const c = model.word_coef[i] * v;
        score += c;
        contributions.push({ term: model.word.terms[i], value: c });
      }
      if (charIndex)
        for (const [i, v] of tfidf(charFeatures(text), charIndex, model.char.idf)) score += model.char_coef[i] * v;
      return { spam: 1 / (1 + Math.exp(-score)), contributions };
    }

    return { classify };
  }

  if (typeof module !== "undefined" && module.exports) module.exports = { create };
  else root.SpamModel = { create };
})(typeof window !== "undefined" ? window : globalThis);
