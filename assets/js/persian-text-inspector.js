(function (root) {
  'use strict';

  const bidiControls = new Set([0x061c, 0x200e, 0x200f, 0x202a, 0x202b, 0x202c, 0x202d, 0x202e, 0x2066, 0x2067, 0x2068, 0x2069]);
  const names = new Map([
    [0x200c, 'ZERO WIDTH NON-JOINER (ZWNJ)'], [0x200d, 'ZERO WIDTH JOINER (ZWJ)'],
    [0x200b, 'ZERO WIDTH SPACE'], [0x00a0, 'NO-BREAK SPACE'], [0x202f, 'NARROW NO-BREAK SPACE'],
    [0x061c, 'ARABIC LETTER MARK (bidi control)'], [0x200e, 'LEFT-TO-RIGHT MARK (bidi control)'],
    [0x200f, 'RIGHT-TO-LEFT MARK (bidi control)'], [0xfeff, 'ZERO WIDTH NO-BREAK SPACE / BOM'],
    [0x064a, 'ARABIC LETTER YEH'], [0x06cc, 'ARABIC LETTER FARSI YEH'],
    [0x0643, 'ARABIC LETTER KAF'], [0x06a9, 'ARABIC LETTER KEHEH'],
    [0x064b, 'ARABIC FATHATAN'], [0x064c, 'ARABIC DAMMATAN'], [0x064d, 'ARABIC KASRATAN'],
    [0x064e, 'ARABIC FATHA'], [0x064f, 'ARABIC DAMMA'], [0x0650, 'ARABIC KASRA'],
    [0x0651, 'ARABIC SHADDA'], [0x0652, 'ARABIC SUKUN'], [0x0640, 'ARABIC TATWEEL']
  ]);
  const persianDigits = '۰۱۲۳۴۵۶۷۸۹';

  function describe(codePoint) {
    if (names.has(codePoint)) return names.get(codePoint);
    if (bidiControls.has(codePoint)) return 'Bidirectional formatting control';
    if ((codePoint >= 0x064b && codePoint <= 0x065f) || codePoint === 0x0670) return 'Arabic diacritic mark';
    if (codePoint >= 0x0660 && codePoint <= 0x0669) return 'Arabic-Indic digit';
    if (codePoint >= 0x06f0 && codePoint <= 0x06f9) return 'Extended Arabic-Indic (Persian) digit';
    if (codePoint >= 0x30 && codePoint <= 0x39) return 'Latin digit';
    if (codePoint === 0x20 || codePoint === 0x09 || codePoint === 0x0a || codePoint === 0x0d) return 'Whitespace';
    if (codePoint < 0x20 || (codePoint >= 0x7f && codePoint <= 0x9f)) return 'Control character';
    return 'Unicode character';
  }

  function inspectText(input, options = {}) {
    let output = String(input);
    const changes = [];
    const apply = (label, transform) => {
      const next = transform(output);
      if (next !== output) changes.push(label);
      output = next;
    };
    if (options.nfc) apply('Unicode NFC', value => value.normalize('NFC'));
    if (options.letters) apply('Arabic Yeh/Kaf to Persian Yeh/Kaf', value => value.replace(/[\u064a\u0643]/g, ch => ch === '\u064a' ? 'ی' : 'ک'));
    if (options.digits) apply('Digits to Persian digits', value => value.replace(/[0-9٠-٩۰-۹]/g, digit => {
      const cp = digit.codePointAt(0);
      const numeric = cp >= 0x0660 && cp <= 0x0669 ? cp - 0x0660 : cp >= 0x06f0 && cp <= 0x06f9 ? cp - 0x06f0 : cp - 0x30;
      return persianDigits[numeric];
    }));
    if (options.bidi) apply('Remove selected bidirectional controls', value => Array.from(value).filter(ch => !bidiControls.has(ch.codePointAt(0))).join(''));

    const counts = new Map();
    for (const character of Array.from(String(input))) {
      const cp = character.codePointAt(0);
      counts.set(cp, (counts.get(cp) || 0) + 1);
    }
    const characters = [...counts.entries()].map(([codePoint, count]) => ({
      character: String.fromCodePoint(codePoint), codePoint,
      hex: `U+${codePoint.toString(16).toUpperCase().padStart(4, '0')}`,
      name: describe(codePoint), count
    }));
    const segmenter = typeof Intl !== 'undefined' && Intl.Segmenter ? new Intl.Segmenter(undefined, { granularity: 'grapheme' }) : null;
    return {
      output, changes, characters,
      codePoints: Array.from(input).length,
      utf16Units: String(input).length,
      graphemes: segmenter ? [...segmenter.segment(String(input))].length : null,
      words: String(input).trim() ? String(input).trim().split(/\s+/u).length : 0
    };
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { inspectText, describe, bidiControls };
  if (!root || !root.document) return;
  const app = root.document.querySelector('[data-text-inspector]');
  if (!app) return;
  const input = app.querySelector('[data-input]');
  const output = app.querySelector('[data-output]');
  const status = app.querySelector('[data-status]');
  const results = app.querySelector('[data-results]');
  const summary = app.querySelector('[data-summary]');
  const characterRows = app.querySelector('[data-characters]');
  const changesList = app.querySelector('[data-changes]');
  const changesSection = app.querySelector('[data-changes-section]');
  const lang = root.document.documentElement.lang;
  const messages = lang === 'fa'
    ? { empty: 'برای شروع، متنی وارد کنید.', done: 'بررسی کامل شد.', copied: 'خروجی کپی شد.', copyFailed: 'کپی خودکار ممکن نشد؛ خروجی را دستی انتخاب کنید.', truncated: 'برای خوانایی، جدول به ۵۰۰ نویسهٔ یکتای نخست محدود شده است.' }
    : { empty: 'Enter text to inspect.', done: 'Inspection complete.', copied: 'Output copied.', copyFailed: 'Automatic copy was unavailable; select the output manually.', truncated: 'The table is limited to the first 500 unique characters for readability.' };
  const formatNumber = value => new Intl.NumberFormat(lang === 'fa' ? 'fa-IR' : 'en').format(value);

  function run() {
    const text = input.value;
    if (!text) { status.textContent = messages.empty; results.hidden = true; return; }
    const options = Object.fromEntries([...app.querySelectorAll('[data-transform]')].map(box => [box.dataset.transform, box.checked]));
    const report = inspectText(text, options);
    output.value = report.output;
    summary.replaceChildren();
    const pairs = [
      [lang === 'fa' ? 'نویسه‌های Unicode' : 'Unicode code points', report.codePoints],
      [lang === 'fa' ? 'واحدهای UTF-16' : 'UTF-16 code units', report.utf16Units],
      [lang === 'fa' ? 'واژه‌ها (تقریبی)' : 'Whitespace-delimited words', report.words]
    ];
    if (report.graphemes !== null) pairs.push([lang === 'fa' ? 'خوشه‌های نویسه‌ای' : 'Grapheme clusters', report.graphemes]);
    for (const [label, value] of pairs) {
      const dt = root.document.createElement('dt'); dt.textContent = label;
      const dd = root.document.createElement('dd'); dd.textContent = formatNumber(value);
      summary.append(dt, dd);
    }
    changesList.replaceChildren();
    for (const change of report.changes) { const li = root.document.createElement('li'); li.textContent = change; changesList.append(li); }
    changesSection.hidden = report.changes.length === 0;
    characterRows.replaceChildren();
    for (const item of report.characters.slice(0, 500)) {
      const row = root.document.createElement('tr');
      let visible = item.character;
      if (item.character === ' ') visible = '␠';
      else if (item.character === '\n') visible = '↵';
      else if (item.character === '\t') visible = '⇥';
      else if (item.character === '\r') visible = '␍';
      else if (bidiControls.has(item.codePoint)) visible = `⟪${item.hex}⟫`;
      else if ([0x200c, 0x200d, 0x200b, 0xfeff, 0x00a0, 0x202f].includes(item.codePoint)) visible = `⟪${item.name.split(' ')[0]}⟫`;
      for (const textValue of [visible, item.hex, item.name, formatNumber(item.count)]) {
        const cell = root.document.createElement('td'); cell.textContent = textValue; row.append(cell);
      }
      characterRows.append(row);
    }
    status.textContent = report.characters.length > 500 ? `${messages.done} ${messages.truncated}` : messages.done;
    results.hidden = false;
  }
  app.querySelector('[data-analyze]').addEventListener('click', run);
  app.querySelector('[data-reset]').addEventListener('click', () => { input.value = ''; output.value = ''; results.hidden = true; status.textContent = ''; input.focus(); });
  app.querySelector('[data-copy]').addEventListener('click', async () => {
    try { await root.navigator.clipboard.writeText(output.value); status.textContent = messages.copied; }
    catch (_) { output.focus(); output.select(); status.textContent = messages.copyFailed; }
  });
})(typeof window !== 'undefined' ? window : null);
