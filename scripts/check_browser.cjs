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
          const toggle = $('.header-controls').getBoundingClientRect();
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
        assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
        await page.locator('#theme').click();
        assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
        await page.reload();
        assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
        await page.locator('#theme').click();
        assert.equal(await page.locator('html').getAttribute('data-theme'),'light');
        assert.equal(facts.lang, lang); assert.equal(facts.dir, lang === 'fa' ? 'rtl' : 'ltr');
        assert.equal(facts.version, 'bilingual-4');
        assert(!facts.overflow, width + ' ' + lang + ' overflow');
        assert(!facts.intersects, width + ' ' + lang + ' nav overlap');
        assert(facts.switchGap > 0, 'language toggle spacing');
        for (const line of [facts.switchUnderline, facts.buttonUnderline, facts.cardUnderline]) assert.equal(line, 'none');
        assert.equal(facts.navLinks, 5); assert.equal(facts.ctas, 3); assert(facts.rss);
        assert.deepEqual(facts.sections, [['writing',3],['start',5],['elsewhere',3],['community',2]]);
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
    const articleRoute = '/fa/writing/document-parsing/';
    for (const route of ['/', '/fa/', '/writing/', '/fa/writing/']) {
      assert((await p.goto(origin+route)).ok());
      assert.equal(await p.locator('a[href="'+articleRoute+'"]').count(), route.startsWith('/fa/') ? 1 : 0);
    }
    for (const width of [375, 1440]) {
      const ctx = await browser.newContext({viewport:{width,height:900},colorScheme:'dark',acceptDownloads:true});
      const a = await ctx.newPage();
      const failures = [];
      a.on('pageerror', err => failures.push(err.message));
      assert((await a.goto(origin+articleRoute)).ok());
      await a.evaluate(()=>document.fonts.ready);
      assert.equal(await a.locator('html').getAttribute('lang'),'fa');
      assert.equal(await a.locator('html').getAttribute('dir'),'rtl');
      assert.equal(await a.locator('main svg').count(),13);
      assert.equal(await a.locator('main table').count(),24);
      assert.equal(await a.locator('.issue-table tbody tr').count(),37);
      assert.equal(await a.locator('.notebook textarea').count(),10);
      assert(await a.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1), 'article horizontal overflow '+width);
      assert(await a.evaluate(()=>[...document.fonts].some(f=>f.family==='Vazirmatn'&&f.status==='loaded')));
      assert.equal(await a.locator('html').getAttribute('data-theme'),'light');
      await a.locator('#theme').click();
      assert.equal(await a.locator('html').getAttribute('data-theme'),'dark');
      await a.locator('[data-filter="open"]').click();
      assert.equal(await a.locator('.issue-table tbody tr:visible').count(),19);
      await a.locator('[data-filter="fixed"]').click();
      assert.equal(await a.locator('.issue-table tbody tr:visible').count(),18);
      await a.locator('[data-filter="all"]').click();
      assert.equal(await a.locator('.issue-table tbody tr:visible').count(),37);
      assert.equal(await a.locator('#overall').textContent(),'99.80%');
      assert.equal(await a.locator('#sensitive').textContent(),'83.33%');
      await a.locator('#errors').evaluate(el=>{el.value='0';el.dispatchEvent(new Event('input',{bubbles:true}));});
      assert.equal(await a.locator('#overall').textContent(),'100.00%');
      assert.equal(await a.locator('#sensitive').textContent(),'100.00%');
      await a.locator('#note0').fill('یادداشت آزمایشی');
      await a.reload();
      assert.equal(await a.locator('#note0').inputValue(),'یادداشت آزمایشی');
      const [download] = await Promise.all([a.waitForEvent('download'),a.locator('#exportNotes').click()]);
      assert.equal(download.suggestedFilename(),'weekend-notes-fa.txt');
      assert.deepEqual(failures,[]);
      await a.screenshot({path:out+'/document-parsing-'+width+'.png',fullPage:false});
      await ctx.close();
    }
    console.log('Verified article discovery, diagrams, tables, RTL, theme, issue filtering, simulator and notes export.');
    console.log('Verified '+count+' locale / viewport / theme browser combinations and 6 localized resource routes.');
  } finally { await browser.close(); }
})().catch(err => { console.error(err); process.exitCode = 1; });
