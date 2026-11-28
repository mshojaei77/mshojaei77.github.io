(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (typeof document !== 'undefined') api.attach(document);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  function segment(text, strategy) {
    const units = [];
    if (strategy === 'characters') {
      let offset = 0;
      for (const value of text) { units.push({ start: offset, end: offset + value.length }); offset += value.length; }
      return units;
    }
    if (strategy === 'words') {
      const regex = /\s*\S+/gu; let match;
      while ((match = regex.exec(text))) units.push({ start: match.index, end: regex.lastIndex });
    } else if (strategy === 'sentences') {
      const regex = /\s*[^.!?؟۔]+(?:[.!?؟۔]+|$)/gu; let match;
      while ((match = regex.exec(text))) units.push({ start: match.index, end: regex.lastIndex });
    } else if (strategy === 'paragraphs') {
      const parts = text.split(/(\r?\n[ \t]*\r?\n+)/);
      let offset = 0;
      for (let i = 0; i < parts.length; i += 2) {
        const value = (parts[i] || '') + (parts[i + 1] || '');
        if (value) units.push({ start: offset, end: offset + value.length });
        offset += value.length;
      }
    } else throw new Error('Choose characters, words, sentences, or paragraphs.');
    if (units.length) units[units.length - 1].end = text.length;
    return units;
  }
  function chunkText(text, strategy = 'words', size = 120, overlap = 20) {
    if (typeof text !== 'string' || !text.trim()) throw new Error('Enter some text to inspect.');
    if (text.length > 200000) throw new Error('Input must be under 200,000 UTF-16 code units.');
    if (!Number.isInteger(size) || size < 1 || size > 2000) throw new Error('Chunk size must be an integer from 1 to 2,000 units.');
    if (!Number.isInteger(overlap) || overlap < 0 || overlap >= size) throw new Error('Overlap must be a non-negative integer smaller than chunk size.');
    const units = segment(text, strategy);
    if (!units.length) throw new Error('Text contains no segmentable units.');
    const chunks = [];
    let startUnit = 0;
    while (startUnit < units.length) {
      const endUnit = Math.min(units.length, startUnit + size);
      const start = units[startUnit].start, end = units[endUnit - 1].end;
      chunks.push({ index: chunks.length + 1, start, end, text: text.slice(start, end), units: endUnit - startUnit, characters: Array.from(text.slice(start, end)).length });
      if (endUnit === units.length) break;
      startUnit = endUnit - overlap;
    }
    const repeatedCharacters = chunks.reduce((sum, chunk) => sum + chunk.characters, 0) - Array.from(text).length;
    return { strategy, unitCount: units.length, chunkCount: chunks.length, sourceCharacters: Array.from(text).length, repeatedCharacters, chunks };
  }
  function attach(doc) {
    const input = doc.querySelector('[data-chunk-input]'), strategy = doc.querySelector('[data-chunk-strategy]'), size = doc.querySelector('[data-chunk-size]'), overlap = doc.querySelector('[data-chunk-overlap]'), run = doc.querySelector('[data-chunk-run]'), output = doc.querySelector('[data-chunk-output]');
    if (!input || !strategy || !size || !overlap || !run || !output) return;
    const fa = doc.documentElement.lang === 'fa';
    run.addEventListener('click', () => {
      try {
        const result = chunkText(input.value, strategy.value, Number(size.value), Number(overlap.value));
        output.dataset.error = 'false';
        output.innerHTML = `<p>${fa ? 'بخش' : 'Chunks'}: ${result.chunkCount} · ${fa ? 'نویسهٔ منبع' : 'Source characters'}: ${result.sourceCharacters} · ${fa ? 'نویسهٔ تکراری' : 'Repeated characters'}: ${result.repeatedCharacters}</p><ol>${result.chunks.map(chunk => `<li><p><strong>#${chunk.index}</strong> · ${fa ? 'بازهٔ UTF-16' : 'UTF-16 offsets'}: ${chunk.start}–${chunk.end} · ${fa ? 'نویسه' : 'characters'}: ${chunk.characters} · ${fa ? 'واحد' : 'units'}: ${chunk.units}</p><pre tabindex="0" dir="auto"><code>${escapeHtml(chunk.text)}</code></pre></li>`).join('')}</ol>`;
      } catch (error) { output.textContent = `${fa ? 'ورودی معتبر نیست: ' : 'Invalid input: '}${error.message}`; output.dataset.error = 'true'; }
    });
    doc.querySelector('[data-chunk-reset]')?.addEventListener('click', () => { input.value = input.dataset.initial || ''; strategy.value = 'words'; size.value = '120'; overlap.value = '20'; output.replaceChildren(); output.dataset.error = 'false'; });
    input.dataset.initial = input.value;
  }
  function escapeHtml(value) { return String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]); }
  return { chunkText, attach };
});
