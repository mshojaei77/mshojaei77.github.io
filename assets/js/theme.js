(function () {
  var root = document.documentElement;
  var key = 'publication-theme';
  function saved() { try { return localStorage.getItem(key); } catch (_) { return null; } }
  function apply(value, persist) {
    var dark = value === 'dark';
    root.dataset.theme = dark ? 'dark' : 'light';
    if (persist) { try { localStorage.setItem(key, root.dataset.theme); } catch (_) {} }
    var button = document.getElementById('theme');
    if (!button) return;
    var fa = root.lang === 'fa';
    var label = dark ? (fa ? 'حالت روشن' : 'Light mode') : (fa ? 'حالت تاریک' : 'Dark mode');
    button.setAttribute('aria-label', label);
    button.setAttribute('aria-pressed', String(dark));
    button.title = label;
    button.textContent = dark ? '☀' : '☾';
  }
  apply(saved() === 'dark' ? 'dark' : 'light', false);
  document.addEventListener('DOMContentLoaded', function () {
    apply(root.dataset.theme, false);
    var button = document.getElementById('theme');
    if (button) button.addEventListener('click', function () { apply(root.dataset.theme === 'dark' ? 'light' : 'dark', true); });
  });
  window.addEventListener('storage', function (e) { if (e.key === key) apply(e.newValue, false); });
})();
