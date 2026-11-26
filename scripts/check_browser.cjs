const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const out = '/tmp/publication-screenshots';
const origin = process.env.SITE_ORIGIN || 'http://127.0.0.1:8765';
const postCounts={en:0,fa:0};for(const name of fs.readdirSync('_posts')){const text=fs.readFileSync('_posts/'+name,'utf8');const lang=text.split(/\r?\n---/)[0].match(/^lang:\s*(en|fa)\s*$/m)?.[1];if(lang)postCounts[lang]++;}
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
        assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
        await page.locator('#site-theme').click();
        assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
        await page.reload();
        assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
        await page.locator('#site-theme').click();
        assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
        const facts = await page.evaluate(() => {
          const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
          const computed = s => getComputedStyle($(s));
          const switcher = $('.language-switch');
          const boxes = [...switcher.children].map(x => x.getBoundingClientRect());
          const nav = $('header nav').getBoundingClientRect();
          const toggle = switcher.getBoundingClientRect();
          const intersects = nav.left < toggle.right && nav.right > toggle.left && nav.top < toggle.bottom && nav.bottom > toggle.top;
          const sections = $$('[data-home-section]').map(s => [s.dataset.homeSection, s.querySelectorAll('.entry-card').length]);
          const grid = $$('[data-home-section="latest"] .entry-card').map(x => x.getBoundingClientRect().top);
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
            sections, grid, startPresent: !!$('.home-start, [data-home-section="start"]'), roadmapLabel: $$('.home-hero .quick-links a')[1].textContent.trim(),
            latest: $$('.home-latest .entry-card h3 a').map(a=>a.getAttribute('href')),
            heroLinks: $$('.home-hero .quick-links a').map(a=>a.getAttribute('href')),
            rss: !!$('footer a[href="/feed.xml"]'),
            internal: $$('main .entry-card a').map(a=>a.getAttribute('href')).filter(x=>x.startsWith('/')),
            og: $('meta[property="og:image"]').content,
            json: JSON.parse($('script[type="application/ld+json"]').textContent)
          };
        });
        assert.equal(facts.lang, lang); assert.equal(facts.dir, lang === 'fa' ? 'rtl' : 'ltr');
        assert.equal(facts.version, 'publication-4');
        assert(!facts.overflow, width + ' ' + lang + ' overflow');
        assert(!facts.intersects, width + ' ' + lang + ' nav overlap');
        assert(facts.switchGap > 0, 'language toggle spacing');
        for (const line of [facts.switchUnderline, facts.buttonUnderline, facts.cardUnderline]) assert.equal(line, 'none');
        assert.equal(facts.navLinks, 3); assert.equal(facts.ctas, 2); assert(facts.rss);
        const latestCount=Math.min(3,postCounts[lang]);
        assert.deepEqual(facts.sections.map(x=>x[0]),latestCount?['latest','elsewhere']:['elsewhere']);
        assert(!facts.startPresent,'Start here must be absent');
        assert.equal(facts.latest.length,latestCount);
        if(width>760&&latestCount>1&&lang==='en')assert.equal(new Set(facts.grid).size,1,'English desktop cards must align');
        if(latestCount>1&&lang==='fa')assert.equal(new Set(facts.grid).size,latestCount,'Persian editorial rows must stack');
        assert.equal(facts.sections.at(-1)[1],4);
        assert.equal(facts.latest.length,new Set(facts.latest).size);
        assert.deepEqual(facts.heroLinks,lang==='fa'?['/fa/writing/','/fa/roadmap/']:['/writing/','/roadmap.html']);
        assert.equal(facts.roadmapLabel,lang==='fa'?'نقشهٔ راه مهندسی هوش مصنوعی':'AI Engineering Roadmap');
        assert(facts.latest.every(x=>x.startsWith(lang==='fa'?'/fa/writing/':'/writing/')));
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
        if(colorScheme==='dark'){await page.locator('#site-theme').click();assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');}
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'theme overflow');
        await page.screenshot({path:out+'/'+lang+'-'+width+'-'+colorScheme+'.png',fullPage:true});
        await page.locator('.language-switch a').click();
        await page.waitForURL(origin+(lang==='en'?'/fa/':'/'));
        if(colorScheme==='dark')await page.locator('#site-theme').click();
        results.push(facts.sections); count++;
      }
      assert(results.every(r=>!r.some(x=>x[0]==='start')));
      await context.close();
    }
    const p = await browser.newPage();
    for(const [route,title] of [['/roadmap.html','AI Engineering Roadmap'],['/fa/roadmap/','نقشهٔ راه مهندسی هوش مصنوعی']]){assert((await p.goto(origin+route)).ok());assert((await p.title()).includes(title));assert.equal(await p.locator('main h1').first().textContent(),title);}
    const isRetired = route => fs.existsSync('retired-posts/'+(route.startsWith('/fa/')?'fa':'en')+'-'+route.split('/').filter(Boolean).at(-1)+'.html');
    const checkSwitch = async (page,route,other,lang) => {
      const fallback=lang==='fa'?'/':'/fa/';
      assert.equal(await page.locator('.language-switch a').getAttribute('href'),isRetired(route)||isRetired(other)?fallback:other);
      if(isRetired(route)||isRetired(other))assert.equal(await page.locator('link[hreflang]').count(),0);
      if(isRetired(route))assert.equal(await page.locator('[data-retired-post]').count(),1);
    };
    for (const [old,route,other,lang] of [["/open_source_ai.html","/writing/open-source-ai/","/fa/writing/open-source-ai/","en"],["/fa/open-source-ai/","/fa/writing/open-source-ai/","/writing/open-source-ai/","fa"],["/resources.html","/writing/resources/","/fa/writing/resources/","en"],["/fa/resources/","/fa/writing/resources/","/writing/resources/","fa"],["/interview.html","/writing/interview/","/fa/writing/interview/","en"],["/fa/interview/","/fa/writing/interview/","/writing/interview/","fa"],["/mini_roadmap.html","/writing/mini-roadmap/","/fa/writing/mini-roadmap/","en"],["/fa/mini-roadmap/","/fa/writing/mini-roadmap/","/writing/mini-roadmap/","fa"],["/materials/Data_Preparation/","/writing/data-preparation/","/fa/writing/data-preparation/","en"],["/fa/materials/Data_Preparation/","/fa/writing/data-preparation/","/writing/data-preparation/","fa"],["/materials/Embeddings/","/writing/embeddings/","/fa/writing/embeddings/","en"],["/fa/materials/Embeddings/","/fa/writing/embeddings/","/writing/embeddings/","fa"],["/materials/Intro/","/writing/intro-to-llms/","/fa/writing/intro-to-llms/","en"],["/fa/materials/Intro/","/fa/writing/intro-to-llms/","/writing/intro-to-llms/","fa"],["/materials/Neural_Networks/","/writing/neural-networks/","/fa/writing/neural-networks/","en"],["/fa/materials/Neural_Networks/","/fa/writing/neural-networks/","/writing/neural-networks/","fa"],["/materials/Tokenization/","/writing/tokenization/","/fa/writing/tokenization/","en"],["/fa/materials/Tokenization/","/fa/writing/tokenization/","/writing/tokenization/","fa"],["/materials/Traditional_LMs/","/writing/traditional-language-models/","/fa/writing/traditional-language-models/","en"],["/fa/materials/Traditional_LMs/","/fa/writing/traditional-language-models/","/writing/traditional-language-models/","fa"],["/materials/Transformers/","/writing/transformers/","/fa/writing/transformers/","en"],["/fa/materials/Transformers/","/fa/writing/transformers/","/writing/transformers/","fa"]]) {
      assert((await p.goto(origin+old)).ok());
      await p.waitForURL(origin+route);
      assert.equal(await p.locator('html').getAttribute('dir'),lang==='fa'?'rtl':'ltr');
      await checkSwitch(p,route,other,lang);
      assert.equal(await p.locator('body').getAttribute('data-layout-version'),'publication-4');
    }
    for(const [old,route,other,lang] of [["/fa/market-analysis/","/fa/writing/market-analysis/","/writing/market-analysis/","fa"],["/market_analysis.html","/writing/market-analysis/","/fa/writing/market-analysis/","en"],["/fa/vector-databases/","/fa/writing/vector-databases/","/writing/vector-databases/","fa"],["/vector_dbs.html","/writing/vector-databases/","/fa/writing/vector-databases/","en"]]){
      assert((await p.goto(origin+old)).ok());
      await p.waitForURL(origin+route);
      assert.equal(await p.locator('html').getAttribute('lang'),lang);
      await checkSwitch(p,route,other,lang);
      assert.equal(await p.locator('header nav a[href="/research/"],header nav a[href="/fa/research/"]').count(),0);
    }
    const articleRoute = '/fa/writing/document-parsing/';
    const englishRoute = '/writing/document-parsing/';
    const faRetired = fs.existsSync('retired-posts/fa-document-parsing.html');
    const enRetired = fs.existsSync('retired-posts/en-document-parsing.html');
    for (const route of ['/', '/fa/', '/writing/', '/fa/writing/']) {
      assert((await p.goto(origin+route)).ok());
      const faCount = await p.locator('a[href="'+articleRoute+'"]').count();
      const enCount = await p.locator('a[href="'+englishRoute+'"]').count();
      if (faRetired) assert.equal(faCount, 0);
      else if (route === '/fa/writing/') assert.equal(faCount, 1);
      else if (route === '/fa/') assert(faCount <= 1);
      else assert.equal(faCount, 0);
      if (enRetired) assert.equal(enCount, 0);
      else if (route === '/writing/') assert.equal(enCount, 1);
      else if (route === '/') assert(enCount <= 1);
      else assert.equal(enCount, 0);
      assert.equal(await p.locator('a[href="/fa/writing/document-parsing-engineering/"]').count(),0);
    }
    for (const lang of enRetired ? [] : ['en']) for (const width of [375, 1440]) {
      const ctx = await browser.newContext({viewport:{width,height:900},colorScheme:'dark',acceptDownloads:true});
      const a = await ctx.newPage();
      const failures = [];
      a.on('pageerror', err => failures.push(err.message));
      assert((await a.goto(origin+(lang==='fa'?articleRoute:englishRoute))).ok());
      await a.evaluate(()=>document.fonts.ready);
      assert.equal(await a.locator('html').getAttribute('lang'),lang);
      assert.equal(await a.locator('html').getAttribute('dir'),lang==='fa'?'rtl':'ltr');
      const expectedSteps = ['contract','setup','native','ocr','tables','layout','structure','rtl','pipeline','evaluation','production','reading'];
      const steps = await a.locator('main h2[data-tutorial-step]').evaluateAll(nodes=>nodes.map(n=>n.dataset.tutorialStep));
      assert.deepEqual(steps, expectedSteps);
      assert.equal(await a.locator('main .article-toc a').count(), expectedSteps.length);
      const ids = await a.locator('main [id]').evaluateAll(nodes=>nodes.map(n=>n.id));
      assert.equal(ids.length, new Set(ids).size);
      for (let i=1;i<=57;i++) assert(ids.includes('section-'+i), 'legacy section anchor '+i);
      assert.equal(await a.locator('main .issue-table, main .notebook, main .sliders').count(),0);
      assert.equal(await a.locator('main pre code').count(),14);
      for(const file of ['requirements.txt','make_demo.py','parse_pdf.py']) {
        const path='/assets/examples/document-parsing/'+file;
        assert.equal(await a.locator('main a[href="'+path+'"][download]').count(),1);
        assert((await ctx.request.get(origin+path)).ok(), path);
      }
      assert(await a.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1), 'article horizontal overflow '+width);
      if (lang==='fa') assert(await a.evaluate(()=>[...document.fonts].some(f=>f.family==='Vazirmatn'&&f.status==='loaded')));
      assert.equal(await a.locator('html').getAttribute('data-theme'),'light');
      await a.locator('#site-theme').click();
      assert.equal(await a.locator('html').getAttribute('data-theme'),'dark');
      await a.reload();
      assert.equal(await a.locator('html').getAttribute('data-theme'),'dark');
      assert.deepEqual(failures,[]);
      await a.screenshot({path:out+'/document-parsing-'+lang+'-'+width+'.png',fullPage:false});
      await a.locator('.language-switch a').click();
      await a.waitForURL(origin+(faRetired?'/fa/':articleRoute));
      await ctx.close();
    }
    for (const width of faRetired ? [] : [375, 1440]) {
      const ctx = await browser.newContext({viewport:{width,height:900}});
      const guide = await ctx.newPage();
      const failures = [];
      guide.on('pageerror', err => failures.push(err.message));
      assert((await guide.goto(origin+articleRoute)).ok());
      await guide.evaluate(()=>document.fonts.ready);
      assert.equal(await guide.locator('html').getAttribute('lang'),'fa');
      assert.equal(await guide.locator('html').getAttribute('dir'),'rtl');
      assert.equal(await guide.locator('main article h1').count(),1);
      assert.equal(await guide.locator('main figure svg').count(),(fs.readFileSync('_posts/2026-11-26-document-parsing.md','utf8').match(/<svg\b/g)||[]).length);
      await checkSwitch(guide,articleRoute,englishRoute,'fa');
      assert.equal(await guide.locator('link[hreflang]').count(),0);
      assert(await guide.evaluate(()=>document.documentElement.scrollWidth <= innerWidth+1), 'engineering guide horizontal overflow '+width);
      assert(await guide.evaluate(()=>[...document.fonts].some(f=>f.family==='Vazirmatn'&&f.status==='loaded')));
      await guide.locator('#site-theme').click();
      assert.equal(await guide.locator('html').getAttribute('data-theme'),'dark');
      assert.deepEqual(failures,[]);
      await guide.screenshot({path:out+'/document-parsing-engineering-fa-'+width+'.png'});
      await ctx.close();
    }
    assert((await p.goto(origin+'/fa/writing/document-parsing-engineering/')).ok());
    await p.waitForURL(origin+articleRoute);
    for(const [route,removed] of [[articleRoute,faRetired],[englishRoute,enRetired]]) if(removed){assert((await p.goto(origin+route)).ok());assert.equal(await p.locator('[data-retired-post]').count(),1);assert.equal(await p.locator('meta[name=robots]').getAttribute('content'),'noindex,nofollow');}
    console.log('Verified tutorial steps, downloadable lab files, legacy anchors, RTL and theme.');
    console.log('Verified '+count+' locale / viewport / theme browser combinations and 26 migrated article redirects.');
  } finally { await browser.close(); }
})().catch(err => { console.error(err); process.exitCode = 1; });
