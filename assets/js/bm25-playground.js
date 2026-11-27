(function (root) {
  'use strict';

  function tokenize(text) {
    return String(text).normalize('NFC').replace(/[\u064a\u0643]/g, ch => ch === '\u064a' ? 'ی' : 'ک').toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  }

  const examples = {
    en: [
      { id: 'E1', text: 'BM25 ranks documents with lexical term frequency and document length.' },
      { id: 'E2', text: 'Hybrid retrieval combines lexical search with embedding vectors.' },
      { id: 'E3', text: 'A retrieval system selects context before language model generation.' }
    ],
    fa: [
      { id: 'F1', text: 'بازیابی اسناد فارسی به نرمال‌سازی متن و ارزیابی دقیق نیاز دارد.' },
      { id: 'F2', text: 'جست‌وجوی واژگانی فراوانی واژه و فراوانی سند را بررسی می‌کند.' },
      { id: 'F3', text: 'بازیابی ترکیبی جست‌وجوی واژگانی و بردارهای embedding را ترکیب می‌کند.' }
    ],
    mixed: [
      { id: 'D1', text: 'BM25 ranks documents with lexical term frequency and document length.' },
      { id: 'D2', text: 'Persian document retrieval needs careful text normalization and evaluation.' },
      { id: 'D3', text: 'Hybrid retrieval combines lexical search with embedding vectors.' },
      { id: 'D4', text: 'A retrieval augmented generation system retrieves context before language model generation.' }
    ]
  };

  function rankDocuments(documents, query, k1 = 1.2, b = 0.75) {
    if (!Array.isArray(documents) || documents.length === 0) throw new Error('Add at least one document.');
    if (documents.length > 100) throw new Error('Use no more than 100 documents in this educational example.');
    if (!Number.isFinite(k1) || k1 < 0 || k1 > 5 || !Number.isFinite(b) || b < 0 || b > 1) throw new Error('k1 must be 0–5 and b must be 0–1.');
    const normalized = documents.map((doc, index) => {
      if (!doc || typeof doc.text !== 'string' || doc.text.length > 10000) throw new Error(`Document ${index + 1} needs text under 10,000 characters.`);
      return { id: String(doc.id ?? `D${index + 1}`), text: doc.text, tokens: tokenize(doc.text) };
    });
    const queryTerms = [...new Set(tokenize(query))];
    const avgdl = normalized.reduce((sum, doc) => sum + doc.tokens.length, 0) / normalized.length;
    const termStats = queryTerms.map(term => {
      const df = normalized.reduce((sum, doc) => sum + (doc.tokens.includes(term) ? 1 : 0), 0);
      const idf = Math.log(1 + (normalized.length - df + 0.5) / (df + 0.5));
      return { term, df, idf };
    });
    const results = normalized.map((doc, order) => {
      const dl = doc.tokens.length;
      const terms = termStats.map(stat => {
        const tf = doc.tokens.filter(token => token === stat.term).length;
        const denominator = tf + k1 * (1 - b + (avgdl ? b * dl / avgdl : 0));
        const contribution = denominator ? stat.idf * (tf * (k1 + 1)) / denominator : 0;
        return { ...stat, tf, denominator, contribution };
      });
      return { id: doc.id, text: doc.text, dl, score: terms.reduce((sum, term) => sum + term.contribution, 0), terms, order };
    }).sort((a, bResult) => bResult.score - a.score || a.order - bResult.order);
    return { results, queryTerms, avgdl, documentCount: normalized.length, k1, b };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { tokenize, rankDocuments, examples };
  if (!root || !root.document) return;
  const app = root.document.querySelector('[data-bm25]');
  if (!app) return;
  const docsInput = app.querySelector('[data-documents]');
  const queryInput = app.querySelector('[data-query]');
  const resultsNode = app.querySelector('[data-results]');
  const rowsNode = app.querySelector('[data-rankings]');
  const summaryNode = app.querySelector('[data-summary]');
  const status = app.querySelector('[data-status]');
  const lang = root.document.documentElement.lang;
  const errorPrefix = lang === 'fa' ? 'ورودی معتبر نیست: ' : 'Invalid input: ';
  const termLabel = lang === 'fa' ? 'واژه' : 'term';
  const f = value => Number(value).toFixed(5);

  function run() {
    try {
      const docs = JSON.parse(docsInput.value);
      const report = rankDocuments(docs, queryInput.value, Number(app.querySelector('[data-k1]').value), Number(app.querySelector('[data-b]').value));
      rowsNode.replaceChildren();
      report.results.forEach((result, index) => {
        const tr = root.document.createElement('tr');
        for (const value of [String(index + 1), result.id, f(result.score)]) { const td = root.document.createElement('td'); td.textContent = value; tr.append(td); }
        const detail = root.document.createElement('td');
        const list = root.document.createElement('ul');
        for (const term of result.terms) {
          const li = root.document.createElement('li');
          li.textContent = `${termLabel} “${term.term}”: tf=${term.tf}, df=${term.df}, idf=${f(term.idf)}, contribution=${f(term.contribution)}`;
          list.append(li);
        }
        detail.append(list); tr.append(detail); rowsNode.append(tr);
      });
      summaryNode.textContent = `${report.documentCount} ${lang === 'fa' ? 'سند' : 'documents'} · ${report.queryTerms.length} ${lang === 'fa' ? 'واژهٔ یکتای پرس‌وجو' : 'unique query terms'} · avgdl=${f(report.avgdl)} · k₁=${report.k1} · b=${report.b}`;
      resultsNode.hidden = false; status.textContent = lang === 'fa' ? 'رتبه‌بندی محاسبه شد.' : 'Ranking calculated.';
    } catch (error) { resultsNode.hidden = true; status.textContent = errorPrefix + error.message; }
  }
  app.querySelector('[data-rank]').addEventListener('click', run);
  app.querySelector('[data-sample]').addEventListener('change', event => {
    const sample = examples[event.target.value] || examples.mixed;
    docsInput.value = JSON.stringify(sample, null, 2);
    queryInput.value = event.target.value === 'fa' ? 'بازیابی اسناد فارسی' : event.target.value === 'en' ? 'retrieval lexical search' : 'retrieval lexical search';
    run();
  });
  app.querySelector('[data-reset]').addEventListener('click', () => {
    app.querySelector('[data-sample]').value = 'mixed';
    docsInput.value = JSON.stringify(examples.mixed, null, 2);
    queryInput.value = 'retrieval lexical search'; app.querySelector('[data-k1]').value = '1.2'; app.querySelector('[data-b]').value = '0.75'; run();
  });
  run();
})(typeof window !== 'undefined' ? window : null);
