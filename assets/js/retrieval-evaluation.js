(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (typeof document !== 'undefined') api.attach(document);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  function evaluateRetrieval(input, k = 5) {
    if (!input || !Array.isArray(input.queries) || !input.queries.length || input.queries.length > 200) throw new Error('Provide 1 to 200 queries.');
    if (!Number.isInteger(k) || k < 1 || k > 1000) throw new Error('K must be an integer from 1 to 1,000.');
    const queries = input.queries.map((query, queryIndex) => {
      if (!query || !Array.isArray(query.ranking) || !query.ranking.length || query.ranking.length > 5000 || !query.relevance || typeof query.relevance !== 'object' || Array.isArray(query.relevance)) throw new Error(`Query ${queryIndex + 1} needs a ranking of 1 to 5,000 documents and a relevance object.`);
      const ranking = [], seen = new Set(); let duplicates = 0;
      for (const id of query.ranking) {
        if (typeof id !== 'string' || !id.trim()) throw new Error(`Query ${queryIndex + 1} has an invalid document ID.`);
        if (seen.has(id)) { duplicates++; continue; }
        seen.add(id); ranking.push(id);
      }
      const relevance = {};
      for (const [id, grade] of Object.entries(query.relevance)) {
        if (!id.trim() || !Number.isInteger(grade) || grade < 0 || grade > 100) throw new Error(`Query ${queryIndex + 1} relevance grades must be integers from 0 to 100.`);
        relevance[id] = grade;
      }
      const judgedRelevant = Object.values(relevance).filter(grade => grade > 0).length;
      const top = ranking.slice(0, k);
      const relevantRetrieved = top.filter(id => (relevance[id] || 0) > 0).length;
      const reciprocalRank = ranking.findIndex(id => (relevance[id] || 0) > 0);
      let precisionSum = 0, dcg = 0, relevantSeen = 0;
      ranking.forEach((id, index) => {
        const grade = relevance[id] || 0;
        if (grade > 0) { relevantSeen++; precisionSum += relevantSeen / (index + 1); }
        if (index < k) dcg += (Math.pow(2, grade) - 1) / Math.log2(index + 2);
      });
      const ideal = Object.values(relevance).filter(grade => grade > 0).sort((a, b) => b - a).slice(0, k);
      const idcg = ideal.reduce((sum, grade, index) => sum + (Math.pow(2, grade) - 1) / Math.log2(index + 2), 0);
      const unjudgedInRanking = ranking.filter(id => !Object.prototype.hasOwnProperty.call(relevance, id)).length;
      return {
        id: String(query.id || `query-${queryIndex + 1}`),
        precisionAtK: relevantRetrieved / k,
        recallAtK: judgedRelevant ? relevantRetrieved / judgedRelevant : 0,
        hitRateAtK: relevantRetrieved ? 1 : 0,
        reciprocalRank: reciprocalRank < 0 ? 0 : 1 / (reciprocalRank + 1),
        dcgAtK: dcg,
        ndcgAtK: idcg ? dcg / idcg : 0,
        averagePrecision: judgedRelevant ? precisionSum / judgedRelevant : 0,
        judgedRelevant,
        unjudgedInRanking,
        duplicates
      };
    });
    const mean = key => queries.reduce((sum, query) => sum + query[key], 0) / queries.length;
    return { k, queries, aggregate: { precisionAtK: mean('precisionAtK'), recallAtK: mean('recallAtK'), hitRateAtK: mean('hitRateAtK'), mrr: mean('reciprocalRank'), ndcgAtK: mean('ndcgAtK'), map: mean('averagePrecision') } };
  }
  function attach(doc) {
    const input = doc.querySelector('[data-eval-input]'), output = doc.querySelector('[data-eval-output]'), kField = doc.querySelector('[data-eval-k]'), run = doc.querySelector('[data-eval-run]');
    if (!input || !output || !kField || !run) return;
    const fa = doc.documentElement.lang === 'fa';
    input.dataset.initial = input.value;
    run.addEventListener('click', () => {
      if (input.value.length > 200000) { output.textContent = fa ? 'اندازهٔ ورودی باید کمتر از ۲۰۰ کیلوبایت باشد.' : 'Input must be under 200 KB.'; output.dataset.error = 'true'; return; }
      try {
        const result = evaluateRetrieval(JSON.parse(input.value), Number(kField.value));
        const n = value => Number(value).toFixed(4);
        output.dataset.error = 'false';
        output.innerHTML = `<h2>${fa ? 'میانگین پرس‌وجوها' : 'Macro average across queries'}</h2><dl class="metric-results"><div><dt>Precision@${result.k}</dt><dd>${n(result.aggregate.precisionAtK)}</dd></div><div><dt>Recall@${result.k}</dt><dd>${n(result.aggregate.recallAtK)}</dd></div><div><dt>Hit Rate@${result.k}</dt><dd>${n(result.aggregate.hitRateAtK)}</dd></div><div><dt>MRR</dt><dd>${n(result.aggregate.mrr)}</dd></div><div><dt>NDCG@${result.k}</dt><dd>${n(result.aggregate.ndcgAtK)}</dd></div><div><dt>MAP</dt><dd>${n(result.aggregate.map)}</dd></div></dl><h3>${fa ? 'جزئیات پرس‌وجوها' : 'Per-query details'}</h3><ul>${result.queries.map(query => `<li><strong>${escapeHtml(query.id)}</strong>: P=${n(query.precisionAtK)}, R=${n(query.recallAtK)}, RR=${n(query.reciprocalRank)}, NDCG=${n(query.ndcgAtK)}, AP=${n(query.averagePrecision)}${query.unjudgedInRanking ? ` · ${fa ? 'بدون برچسب' : 'unjudged'}: ${query.unjudgedInRanking}` : ''}${query.duplicates ? ` · ${fa ? 'تکراری حذف‌شده' : 'duplicates removed'}: ${query.duplicates}` : ''}</li>`).join('')}</ul>`;
      } catch (error) {
        output.dataset.error = 'true'; output.textContent = `${fa ? 'ورودی معتبر نیست: ' : 'Invalid input: '}${error.message}`;
      }
    });
    doc.querySelector('[data-eval-reset]')?.addEventListener('click', () => { input.value = input.dataset.initial || ''; kField.value = '5'; output.replaceChildren(); output.dataset.error = 'false'; });
  }
  function escapeHtml(value) { return String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]); }
  return { evaluateRetrieval, attach };
});
