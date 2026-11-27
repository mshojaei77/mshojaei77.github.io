/* Load only a versioned, integrity-checked GoatCounter script on the production host.
   No API key, account secret, or analytics code in Publication Studio. */
(() => {
  'use strict';
  const config = document.currentScript;
  if (!config || location.protocol !== 'https:' || location.hostname !== config.dataset.productionHost) return;
  if (location.pathname.startsWith('/admin/') || location.pathname === '/admin') return;
  const site = config.dataset.goatcounterSite;
  if (!/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(site || '') || site.length > 63) return;
  const script = document.createElement('script');
  script.src = 'https://gc.zgo.at/count.v5.js';
  script.async = true;
  script.crossOrigin = 'anonymous';
  script.integrity = 'sha384-atnOLvQb9t+jTSipvd75X2yginT4PjVbqDdlJAmxMm+wYElFmeR6EmLP5bYeoRVQ';
  script.dataset.goatcounter = 'https://' + site + '.goatcounter.com/count';
  document.head.appendChild(script);
})();
