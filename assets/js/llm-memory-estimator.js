(function (root) {
  'use strict';

  function estimateMemory(input) {
    const positive = ['parametersBillions', 'weightBits', 'batchSize', 'contextLength', 'layers', 'kvHeads', 'headDimension', 'kvBytes'];
    for (const key of positive) if (!Number.isFinite(input[key]) || input[key] <= 0) throw new Error(`${key} must be a positive finite number.`);
    if (!Number.isInteger(input.batchSize) || !Number.isInteger(input.contextLength) || !Number.isInteger(input.layers) || !Number.isInteger(input.kvHeads) || !Number.isInteger(input.headDimension)) throw new Error('Batch, context, layer, head, and dimension values must be whole numbers.');
    if (!Number.isFinite(input.runtimeOverheadPercent) || input.runtimeOverheadPercent < 0 || input.runtimeOverheadPercent > 500) throw new Error('Runtime overhead must be between 0% and 500%.');
    if (!Number.isFinite(input.safetyMarginPercent) || input.safetyMarginPercent < 0 || input.safetyMarginPercent > 200) throw new Error('Safety margin must be between 0% and 200%.');
    if (![2, 3, 4, 8, 16].includes(input.weightBits)) throw new Error('Weight precision must be one of 2, 3, 4, 8, or 16 bits.');
    if (![1, 2, 4].includes(input.kvBytes)) throw new Error('KV cache element size must be 1, 2, or 4 bytes.');
    const weightBytes = input.parametersBillions * 1e9 * input.weightBits / 8;
    const kvCacheBytes = 2 * input.layers * input.batchSize * input.contextLength * input.kvHeads * input.headDimension * input.kvBytes;
    const baseBytes = weightBytes + kvCacheBytes;
    const runtimeOverheadBytes = baseBytes * input.runtimeOverheadPercent / 100;
    const estimatedBytes = baseBytes + runtimeOverheadBytes;
    const safetyMarginBytes = estimatedBytes * input.safetyMarginPercent / 100;
    const recommendedBytes = estimatedBytes + safetyMarginBytes;
    if (![weightBytes, kvCacheBytes, runtimeOverheadBytes, safetyMarginBytes, recommendedBytes].every(Number.isFinite)) throw new Error('The values exceed the supported numeric range.');
    const sizes = bytes => ({ bytes, GB: bytes / 1e9, GiB: bytes / (1024 ** 3) });
    return { weights: sizes(weightBytes), kvCache: sizes(kvCacheBytes), base: sizes(baseBytes), runtimeOverhead: sizes(runtimeOverheadBytes), estimate: sizes(estimatedBytes), safetyMargin: sizes(safetyMarginBytes), planningTotal: sizes(recommendedBytes) };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { estimateMemory };
  if (!root || !root.document) return;
  const app = root.document.querySelector('[data-memory-estimator]');
  if (!app) return;
  const status = app.querySelector('[data-status]');
  const results = app.querySelector('[data-results]');
  const output = app.querySelector('[data-output]');
  const lang = root.document.documentElement.lang;
  const labels = lang === 'fa'
    ? { weights: 'وزن مدل (تخمین فشرده)', kvCache: 'KV cache', base: 'وزن + KV cache', runtimeOverhead: 'سربار runtime واردشده توسط شما', estimate: 'جمع تخمینی پیش از حاشیهٔ ایمنی', safetyMargin: 'حاشیهٔ ایمنی واردشده توسط شما', planningTotal: 'کل پیشنهادی برای برنامه‌ریزی' }
    : { weights: 'Model weights (packed estimate)', kvCache: 'KV cache', base: 'Weights + KV cache', runtimeOverhead: 'User-entered runtime overhead', estimate: 'Estimated subtotal before safety margin', safetyMargin: 'User-entered safety margin', planningTotal: 'Suggested planning total' };
  const number = value => new Intl.NumberFormat(lang === 'fa' ? 'fa-IR' : 'en', { maximumFractionDigits: 3 }).format(value);
  function read() {
    const value = selector => Number(app.querySelector(selector).value);
    return { parametersBillions: value('[data-param-b]'), weightBits: value('[data-weight-bits]'), batchSize: value('[data-batch]'), contextLength: value('[data-context]'), layers: value('[data-layers]'), kvHeads: value('[data-kv-heads]'), headDimension: value('[data-head-dim]'), kvBytes: value('[data-kv-bytes]'), runtimeOverheadPercent: value('[data-overhead]'), safetyMarginPercent: value('[data-margin]') };
  }
  function run() {
    try {
      const report = estimateMemory(read()); output.replaceChildren();
      for (const key of ['weights', 'kvCache', 'base', 'runtimeOverhead', 'estimate', 'safetyMargin', 'planningTotal']) {
        const dt = root.document.createElement('dt'); dt.textContent = labels[key];
        const dd = root.document.createElement('dd'); dd.textContent = `${number(report[key].GiB)} GiB · ${number(report[key].GB)} GB`;
        output.append(dt, dd);
      }
      results.hidden = false; status.textContent = lang === 'fa' ? 'برآورد محاسبه شد؛ فرض‌ها و محدودیت‌ها را بررسی کنید.' : 'Estimate calculated; review the assumptions and limitations.';
    } catch (error) { results.hidden = true; status.textContent = (lang === 'fa' ? 'ورودی معتبر نیست: ' : 'Invalid input: ') + error.message; }
  }
  app.querySelector('[data-estimate]').addEventListener('click', run);
  app.querySelector('[data-reset]').addEventListener('click', () => {
    app.querySelector('[data-param-b]').value = '7'; app.querySelector('[data-weight-bits]').value = '4';
    app.querySelector('[data-batch]').value = '1'; app.querySelector('[data-context]').value = '4096';
    app.querySelector('[data-layers]').value = '32'; app.querySelector('[data-kv-heads]').value = '8';
    app.querySelector('[data-head-dim]').value = '128'; app.querySelector('[data-kv-bytes]').value = '2';
    app.querySelector('[data-overhead]').value = '15'; app.querySelector('[data-margin]').value = '20'; run();
  });
  run();
})(typeof window !== 'undefined' ? window : null);
