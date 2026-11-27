(function (root) {
  'use strict';
  const schemaVersion = 1;

  function parseState(raw, moduleIds) {
    if (!raw) return { schema_version: schemaVersion, completed: {} };
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || parsed.schema_version !== schemaVersion || !parsed.completed || typeof parsed.completed !== 'object') return { schema_version: schemaVersion, completed: {} };
      const allowed = new Set(moduleIds);
      const completed = {};
      for (const id of allowed) if (parsed.completed[id] === true) completed[id] = true;
      return { schema_version: schemaVersion, completed };
    } catch (_) { return { schema_version: schemaVersion, completed: {} }; }
  }

  function completionCount(state, moduleIds) {
    return moduleIds.reduce((count, id) => count + (state.completed[id] === true ? 1 : 0), 0);
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = { parseState, completionCount, schemaVersion };
  if (!root || !root.document) return;
  const page = root.document.querySelector('[data-path-id]');
  if (!page) return;
  const boxes = [...page.querySelectorAll('[data-module-id]')];
  const ids = boxes.map(box => box.dataset.moduleId);
  const key = `ai-engineering-learning:v${schemaVersion}:${page.dataset.pathId}`;
  const summary = page.querySelector('[data-progress-summary]');
  const reset = page.querySelector('[data-progress-reset]');
  const lang = root.document.documentElement.lang;
  let storage;
  let state;
  try { storage = root.localStorage; state = parseState(storage.getItem(key), ids); }
  catch (_) { storage = null; state = parseState(null, ids); }
  const updateSummary = () => {
    const count = completionCount(state, ids);
    summary.textContent = lang === 'fa' ? `${count} از ${ids.length} بخش در این مرورگر علامت‌گذاری شده است.` : `${count} of ${ids.length} modules marked complete in this browser.`;
  };
  const save = () => {
    try { if (storage) storage.setItem(key, JSON.stringify(state)); }
    catch (_) { storage = null; }
    updateSummary();
  };
  for (const box of boxes) {
    box.checked = state.completed[box.dataset.moduleId] === true;
    box.addEventListener('change', () => {
      if (box.checked) state.completed[box.dataset.moduleId] = true;
      else delete state.completed[box.dataset.moduleId];
      save();
    });
  }
  reset.addEventListener('click', () => {
    state = { schema_version: schemaVersion, completed: {} };
    for (const box of boxes) box.checked = false;
    try { if (storage) storage.removeItem(key); } catch (_) { storage = null; }
    updateSummary();
  });
  updateSummary();
})(typeof window !== 'undefined' ? window : null);
