const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const out = '/tmp/publication-screenshots';
const origin = process.env.SITE_ORIGIN || 'http://127.0.0.1:8765';
fs.mkdirSync(out, { recursive: true });
(async () => {
  const browser = await chromium.launch({ headless: true });
  let count = 0;
  try {
    for (const width of [375, 768, 1024, 1440]) for (const colorScheme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, colorScheme, deviceScaleFactor: 1 });
      const page = await context.newPage();
      const results = [];
      for (const lang of ['en', 'fa']) {
        const route = lang === 'fa' ? '/fa/' : '/';
        const response = await page.goto(origin + route, { waitUntil: 'networkidle' });
        assert(response.ok(), route);
        await page.evaluate(() => document.fonts.ready);
        const facts = await page.evaluate(() => {
          const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
          const computed = s => getComputedStyle($(s));
          const switcher = $('.language-switch');
          const boxes = [...switcher.children].map(x => x.getBoundingClientRect());
          const nav = $('header nav').getBoundingClientRect();
          const toggle = switcher.getBoundingClientRect();
          const intersects = nav.left < toggle.right && nav.right > toggle.left && nav.top < toggle.bottom && nav.bottom > toggle.top;
          const sections = $$('[data-home-section]').map(s => [s.dataset.homeSection, s.querySelectorAll('.entry-card').length]);
          const grid = $$('[data-home-section="writing"] .entry-card').map(x => x.getBoundingClientRect().top);
          return {
            lang: document.documentElement.lang, dir: document.documentElement.dir,
            version: document.body.dataset.layoutVersion, title: document.title,
            overflow: document.documentElement.scrollWidth > innerWidth + 1,
            navLinks: $$('header nav a').length, ctas: $$('.quick-links .action').length,
            switchGap: boxes[1].left - boxes[0].right, intersects,
            switchUnderline: getComputedStyle($('.language-switch a')).textDecorationLine,
            buttonUnderline: computed('.action').textDecorationLine,
            cardUnderline: computed('.entry-card h3 a').textDecorationLine,
            fonts: ['body','h1','.eyebrow','.lede','header nav','.action','.entry-card h3','.entry-card p'].map(s=>computed(s).fontFamily),
            fontLoaded: [...document.fonts].some(f=>f.family === 'Vazirmatn' && f.status === 'loaded'),
            sections, grid, rss: !!$('footer a[href="/feed.xml"]'),
            internal: $$('main .entry-card a').map(a=>a.getAttribute('href')).filter(x=>x.startsWith('/')),
            og: $('meta[property="og:image"]').content,
            json: JSON.parse($('script[type="application/ld+json"]').textContent)
          };
        });
        assert.equal(facts.lang, lang); assert.equal(facts.dir, lang === 'fa' ? 'rtl' : 'ltr');
        assert.equal(facts.version, 'bilingual-3');
        assert(!facts.overflow, width + ' ' + lang + ' overflow');
        assert(!facts.intersects, width + ' ' + lang + ' nav overlap');
        assert(facts.switchGap > 0, 'language toggle spacing');
        for (const line of [facts.switchUnderline, facts.buttonUnderline, facts.cardUnderline]) assert.equal(line, 'none');
        assert.equal(facts.navLinks, 7); assert.equal(facts.ctas, 3); assert(facts.rss);
        assert.deepEqual(facts.sections, [['writing',3],['projects',2],['start',6],['elsewhere',3]]);
        if (width > 680) assert.equal(new Set(facts.grid).size, 1, 'featured cards should share a row');
        if (lang === 'fa') {
          assert(facts.fontLoaded, 'self-hosted Persian font did not load');
          assert(facts.fonts.every(x=>x.startsWith('Vazirmatn')), JSON.stringify(facts.fonts));
          assert(facts.internal.every(x=>x.startsWith('/fa/')), JSON.stringify(facts.internal));
          assert(facts.title.includes('محمد شجاعی | مهندس هوش مصنوعی'));
        }
        assert.equal(facts.json['@graph'][0].inLanguage, lang);
        assert(facts.og.endsWith('/assets/img/social-card.png'));
        const img = await page.evaluate(async () => { const i = new Image(); i.src='/assets/img/social-card.png'; await i.decode(); return [i.naturalWidth,i.naturalHeight]; });
        assert.deepEqual(img, [1200,630]);
        await page.screenshot({path:out+'/'+lang+'-'+width+'-'+colorScheme+'.png',fullPage:true});
        await page.locator('.language-switch a').click();
        await page.waitForURL(origin+(lang==='en'?'/fa/':'/'));
        results.push(facts.sections); count++;
      }
      assert.deepEqual(results[0], results[1]);
      await context.close();
    }
    const p = await browser.newPage();
    for (const [slug, original] of [['open-source-ai','open_source_ai.html'],['vector-databases','vector_dbs.html'],['market-analysis','market_analysis.html'],['resources','resources.html'],['interview','interview.html'],['mini-roadmap','mini_roadmap.html']]) {
      assert((await p.goto(origin+'/fa/'+slug+'/')).ok());
      assert.equal(await p.locator('html').getAttribute('dir'),'rtl');
      assert.equal(await p.locator('.language-switch a').getAttribute('href'),'/'+original);
      assert.equal(await p.locator('main a.action').getAttribute('href'),'/'+original);
    }
    console.log('Verified '+count+' locale / viewport / theme browser combinations and 6 localized resource routes.');
  } finally { await browser.close(); }
})().catch(err => { console.error(err); process.exitCode = 1; });
