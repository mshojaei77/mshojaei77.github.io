const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {createHash}=require('node:crypto');
const origin=process.env.SITE_ORIGIN||'http://127.0.0.1:8765';
const out='/tmp/publication-screenshots';
const path='_posts/2026-11-26-document-parsing.md';
const en='_posts/2026-11-26-document-parsing-en.html';
const repo='/repos/mshojaei77/mshojaei77.github.io';
const fixture=(lang,route,body)=>['---','layout: article','lang: '+lang,'title: "Test article"','description: "Test description"','excerpt: "Test description"','date: 2026-11-26 00:00:00 +0000','permalink: '+route,'---',body].join('\n');
const source=fs.existsSync(path)?fs.readFileSync(path,'utf8'):fixture('fa','/fa/writing/document-parsing/','## سلام\n'+'<figure><svg xmlns="http://www.w3.org/2000/svg"></svg></figure>\n'.repeat(25)+'متن فارسی '.repeat(28000));
const sha=b=>createHash('sha1').update('blob '+b.length+'\0').update(b).digest('hex');
const entry=(p,b)=>({path:p,name:p.split('/').pop(),type:'file',encoding:'base64',size:b.length,sha:sha(b),content:b.toString('base64')});
const main=new Map([[path,Buffer.from(source)],[en,fs.existsSync(en)?fs.readFileSync(en):Buffer.from(fixture('en','/writing/document-parsing/','<p>Test</p>'))]]);
const trees=new Map(),commits=new Map(),branches=new Map(),blobs=new Map(),prs=new Map();
let denied=0,writeCount=0,prCreates=0,loseResponse=true,mainChanged=false,treeSeq=0,commitSeq=0,prSeq=0;
let mainRef='a'.repeat(40);let commitCount=0,readCount=0,casConflict=false,confirmDelete=true;
const headers={'access-control-allow-origin':'*','access-control-allow-methods':'GET,POST,OPTIONS','access-control-allow-headers':'authorization,content-type,x-github-api-version,accept','content-type':'application/json'};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const browser=await chromium.launch({headless:true});try{
 const ctx=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true});
 const page=await ctx.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('dialog',x=>!confirmDelete&&x.message().includes('Delete this post')?x.dismiss():x.accept());
 await ctx.route('https://api.github.com/**',async route=>{
  const req=route.request(),u=new URL(req.url()),p=decodeURIComponent(u.pathname),method=req.method(),token=req.headers().authorization;
  const reply=(status,json)=>route.fulfill({status,headers,body:JSON.stringify(json)});
  if(method==='OPTIONS')return route.fulfill({status:204,headers});
  if(p==='/user'){if(token==='Bearer outsider'){denied++;return reply(200,{id:99,login:'other'});}if(token!=='Bearer github_'+'pat_studio_test')return reply(401,{});return reply(200,{id:76538971,login:'mshojaei77'});}
  if(token!=='Bearer github_'+'pat_studio_test')return reply(401,{});
  if(p===repo)return reply(200,{full_name:'mshojaei77/mshojaei77.github.io',owner:{id:76538971},permissions:{push:true}});
  if(p===repo+'/contents/_posts')return reply(200,[...main].map(([p,b])=>({path:p,type:'file',sha:sha(b)})));
  if(p.startsWith(repo+'/contents/')){readCount++;const name=p.slice((repo+'/contents/').length),ref=u.searchParams.get('ref'),files=ref==='main'||ref===mainRef?main:commits.get(ref);let b=files?.get(name);if(mainChanged&&name===path&&(ref==='main'||ref===mainRef))b=Buffer.from(source+'\nchanged externally');return b?reply(200,entry(name,b)):reply(404,{});}
  if(p.startsWith(repo+'/git/ref/heads/')){const name=p.slice((repo+'/git/ref/heads/').length);const c=name==='main'?mainRef:branches.get(name);return c?reply(200,{object:{sha:c}}):reply(404,{});}
  if(p===repo+'/git/commits/'+mainRef)return reply(200,{tree:{sha:'base-tree'}});
  if(p.startsWith(repo+'/pulls/')&&method==='GET'){const pr=[...prs.values()].find(x=>x.number===Number(p.split('/').pop()));return pr?reply(200,pr):reply(404,{});}
  if(p===repo+'/pulls'&&method==='GET'){const name=u.searchParams.get('head').split(':').slice(1).join(':');return reply(200,prs.has(name)?[prs.get(name)]:[]);}
  if(p==='/graphql'&&method==='POST'){
    writeCount++;const input=req.postDataJSON().variables.input;
    assert.equal(input.branch.branchName,'main');assert.equal(input.branch.repositoryNameWithOwner,'mshojaei77/mshojaei77.github.io');
    if(casConflict){casConflict=false;mainRef='b'.repeat(40);return reply(200,{errors:[{message:'Expected head oid does not match'}]});}
    if(input.expectedHeadOid!==mainRef)return reply(200,{errors:[{message:'Expected head oid does not match'}]});
    for(const f of input.fileChanges.deletions){assert(main.has(f.path));main.delete(f.path);}
    for(const f of input.fileChanges.additions)main.set(f.path,Buffer.from(f.contents,'base64'));
    commitCount++;mainRef=String(commitCount).padStart(40,'0');
    if(loseResponse){loseResponse=false;return route.abort('failed');}
    return reply(200,{data:{createCommitOnBranch:{commit:{oid:mainRef,url:'https://github.com/mshojaei77/mshojaei77.github.io/commit/'+mainRef}}}});
  }
  if(method!=='POST')return reply(404,{});
  writeCount++;const body=req.postDataJSON();
  if(p===repo+'/git/blobs'){const b=Buffer.from(body.content,'base64'),id=sha(b);blobs.set(id,b);return reply(201,{sha:id});}
  if(p===repo+'/git/trees'){assert.equal(body.base_tree,'base-tree');const files=new Map(main);for(const x of body.tree){assert(x.path.startsWith('_posts/')||x.path.startsWith('assets/uploads/')||x.path.startsWith('retired-posts/'));if(x.sha===null)files.delete(x.path);else files.set(x.path,x.content===undefined?blobs.get(x.sha):Buffer.from(x.content));}const id='tree-'+(++treeSeq);trees.set(id,files);return reply(201,{sha:id});}
  if(p===repo+'/git/commits'){assert.deepEqual(body.parents,[mainRef]);const id='commit-'+(++commitSeq);commits.set(id,trees.get(body.tree));return reply(201,{sha:id});}
  if(p===repo+'/git/refs'){assert(body.ref.startsWith('refs/heads/studio/'));const name=body.ref.slice('refs/heads/'.length);assert(!branches.has(name));branches.set(name,body.sha);return reply(201,{object:{sha:body.sha}});}
  if(p===repo+'/pulls'){assert.equal(body.base,'main');assert(branches.has(body.head));assert(!prs.has(body.head));prCreates++;const pr={number:++prSeq,state:'open'};prs.set(body.head,pr);if(loseResponse){loseResponse=false;return route.abort('failed');}return reply(201,pr);}
  throw Error('Unexpected mutation '+p);
 });
 async function signIn(token='github_'+'pat_studio_test'){await page.locator('#token').fill(token);await page.locator('#login-form button[type=submit]').click();}
 const waitStatus=async re=>{await page.waitForFunction(s=>new RegExp(s,'i').test(document.querySelector('#status').dataset.message),re);};
 assert((await page.goto(origin+'/admin/')).ok());await page.locator('#login').waitFor();
 await signIn('outsider');await page.getByText('Access denied: this workspace is reserved for its owner.').waitFor();assert.equal(denied,1);assert.equal(writeCount,0);assert(await page.locator('#workspace').isHidden());
 await page.locator('#remember').check();await signIn();await page.locator('#workspace').waitFor();await page.locator('#dashboard-title').waitFor();assert.equal(await page.locator('#dash-published').textContent(),'1');assert.equal(await page.locator('#dash-drafts').textContent(),'0');assert.equal(await page.locator('#dashboard-posts .dashboard-row').count(),1);assert.equal(await page.locator('#dashboard-posts .dashboard-row').getAttribute('data-path'),en);assert.equal(await page.locator('#dashboard-home').getAttribute('aria-current'),'page');assert.equal(await page.locator('#dashboard-provider-links .dashboard-provider').count(),3);assert.equal(await page.locator('#dashboard-provider-links a[href^="https://mshojaei77.goatcounter.com/"]').count(),1);assert.equal(await page.locator('#dashboard-provider-links a[href^="https://search.google.com/search-console?"]').count(),1);assert.equal(await page.locator('#dashboard-provider-links a[href^="https://www.bing.com/webmasters/"]').count(),1);assert.equal(await page.locator('#token').inputValue(),'');assert(await page.evaluate(()=>JSON.parse(localStorage.getItem('studio-remember-v1')).expires>Date.now()));
 assert.equal(await page.locator('html').getAttribute('lang'),'en');assert.equal(await page.locator('.doc[data-path="'+path+'"]').count(),0);assert.equal(await page.locator('.doc[data-path="'+en+'"]').count(),1);
 await page.locator('#locale-fa').click();await page.locator('.doc[data-path="'+path+'"]').waitFor();assert.equal(await page.locator('html').getAttribute('dir'),'rtl');assert.equal(await page.locator('.doc[data-path="'+en+'"]').count(),0);assert.equal(await page.locator('.doc[data-path="about.md"]').count(),0);assert.equal(await page.locator('.doc[data-path="fa/about/index.html"]').count(),1);
 assert.equal(await page.locator('#dashboard-posts .dashboard-row').getAttribute('data-path'),path);await page.locator('#dash-filter-drafts').click();assert.equal(await page.locator('#dashboard-posts .dashboard-row').count(),0);assert(await page.locator('#dashboard-no-posts').isVisible());await page.locator('#dash-filter-all').click();const openTime=Date.now();await page.locator('.dashboard-row[data-path="'+path+'"] .dash-edit').click();await page.locator('#editor-pane').waitFor();await waitStatus('Loaded from GitHub');assert(await page.locator('#post-date').isVisible());assert(await page.locator('#post-date').isEnabled());assert.equal(await page.locator('#metadata').getAttribute('open'),null);assert(await page.locator('#post-date').evaluate(el=>document.activeElement===el));
 const body=source.replace(/^(?:\uFEFF)?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/,'');assert.equal(await page.locator('#source').inputValue(),body);assert.equal(await page.locator('#source').getAttribute('dir'),'rtl');
 assert(await page.locator('#preview').isHidden());
 const times=await page.locator('#source').evaluate(el=>{const times=[];const original=el.value;for(let i=0;i<20;i++){const t=performance.now();el.setRangeText('x',el.value.length,el.value.length,'end');el.dispatchEvent(new Event('input'));times.push(performance.now()-t);}el.value=original;el.dispatchEvent(new Event('input'));return times.sort((a,b)=>a-b);});
 const p95=times[Math.ceil(times.length*.95)-1];assert(p95<100,'Editor input p95 '+p95+' ms');console.log('Studio actual manuscript bytes='+Buffer.byteLength(source)+' mocked open ms='+(Date.now()-openTime)+' synchronous input event p95 ms='+p95.toFixed(2));
 await page.locator('#preview-toggle').click();await page.frameLocator('#preview').locator('svg').first().waitFor();assert.equal(await page.frameLocator('#preview').locator('svg').count(),(body.match(/<svg\b/g)||[]).length);
 await page.locator('#preview-toggle').click();const marker='\n<!-- Studio test edit -->\n';await page.locator('#source').evaluate((el,m)=>{el.setRangeText(m,el.value.length,el.value.length,'end');el.dispatchEvent(new Event('input'));},marker);
 await page.locator('#dashboard-home').click();await page.locator('#empty').waitFor();assert.equal(await page.locator('#dash-drafts').textContent(),'1');await page.locator('#dash-filter-drafts').click();assert.equal(await page.locator('#dashboard-posts .dashboard-row').count(),1);await page.locator('#locale-en').click();await page.locator('#empty').waitFor();assert.equal(await page.locator('.doc[data-path="'+path+'"]').count(),0);assert.equal(await page.locator('.doc[data-path="'+en+'"]').count(),1);await page.locator('#locale-fa').click();await page.locator('.doc[data-path="'+path+'"]').click();await waitStatus('Recovered local draft');assert.equal(await page.locator('#source').inputValue(),body+marker);await page.locator('#local-save').click();await waitStatus('Draft saved on this device');assert.equal(writeCount,0);
 const expiry=await page.evaluate(()=>JSON.parse(localStorage.getItem('studio-remember-v1')).expires);await page.reload();await page.locator('#workspace').waitFor();assert.equal(await page.locator('html').getAttribute('lang'),'fa');assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('studio-remember-v1')).expires),expiry);await page.locator('.doc[data-path="'+path+'"]').click();await waitStatus('Recovered local draft');assert.equal(await page.locator('#source').inputValue(),body+marker);
 const revisedDay='2025-02-13';const renamed='عنوان تازهٔ آزمایشی';assert(await page.locator('#post-date').isVisible());assert(await page.locator('#post-date').isEnabled());assert.equal(await page.locator('#post-date').inputValue(),source.match(/^date:\s*(\d{4}-\d{2}-\d{2})/m)[1]);await page.locator('#post-date').fill('2099-01-01');await waitStatus('Choose a valid publication date');await page.locator('#post-date').fill(revisedDay);assert.equal(await page.locator('.doc[data-path="'+path+'"] time').textContent(),revisedDay);await page.locator('#metadata summary').click();await page.locator('#title').fill(renamed);assert.equal(await page.locator('.doc[data-path="'+path+'"] span').textContent(),renamed);await page.locator('#metadata summary').click();
 mainChanged=true;await page.locator('#submit').click();await waitStatus('Main has changed');assert.equal(writeCount,0);mainChanged=false;
 casConflict=true;await page.locator('#submit').click();await waitStatus('Expected head oid');assert.equal(commitCount,0);
 await page.locator('#submit').click();await waitStatus('network request failed');assert.equal(commitCount,1);const after=writeCount;
 await page.locator('#submit').click();await waitStatus('Committed to GitHub');assert.equal(commitCount,1);assert.equal(writeCount,after);
 assert.equal(main.get(path).toString('utf8'),source.replace(/^title:.*$/m,'title: '+JSON.stringify(renamed)).replace(/^date:.*$/m,'date: '+JSON.stringify(revisedDay+' 00:00:00 +0000'))+marker);assert(!(await page.locator('#source').evaluate(e=>e.readOnly)));
 assert.equal(await page.locator('#commit-link').getAttribute('href'),'https://github.com/mshojaei77/mshojaei77.github.io/commit/'+mainRef);
 assert.equal(await page.locator('.doc[data-path="'+path+'"] span').textContent(),renamed);await page.reload();await page.locator('#workspace').waitFor();await page.locator('.doc[data-path="'+path+'"] span').filter({hasText:renamed}).waitFor();assert.equal(await page.locator('.doc[data-path="'+path+'"] time').textContent(),revisedDay);
 await page.locator('#new').click();await page.locator('#new-title').fill('آزمایش جدید');await page.locator('#new-date').fill('2025-03-01');await page.locator('#new-slug').fill('studio-test-post');await page.locator('#new-description').fill('توضیح آزمایشی');await page.locator('#new-form button[type=submit]').click();await waitStatus('Local draft created');assert.equal(writeCount,after);
 await page.locator('#source').fill('## Hello\n\n**سلام**');const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aawAAAABJRU5ErkJggg==','base64');await page.locator('#media').setInputFiles({name:'pixel.png',mimeType:'image/png',buffer:png});await waitStatus('Image staged locally');
 const readsBefore=readCount;
 await page.locator('#submit').click();await waitStatus('Committed to GitHub');assert.equal(commitCount,2);assert(readCount-readsBefore<=3,'new article must not fetch every existing post');
 const newPath=[...main.keys()].find(x=>x.includes('studio-test-post'));const asset=[...main.keys()].find(x=>x.startsWith('assets/uploads/'));assert(newPath&&asset);assert(newPath.startsWith('_posts/2025-03-01-'));assert(main.get(newPath).toString().includes('date: 2025-03-01 00:00:00 +0000'));assert(main.get(newPath).toString().includes('/fa/writing/studio-test-post/'));assert(main.get(newPath).toString().includes('/'+asset));assert.deepEqual(main.get(asset),png);

 // Import actual Markdown into a new local draft, preserve source including SVG/Unicode.
 const importBody='# فایل واردشده\n\n**سلام**\n<figure><svg></svg></figure>\n';
 await page.locator('#import-new').click();
 await page.locator('#import-file').setInputFiles({name:'imported.md',mimeType:'text/markdown',buffer:Buffer.from(importBody)});
 await page.locator('#new-heading').waitFor();
 assert.equal(await page.locator('#new-title').inputValue(),'فایل واردشده');
 assert.equal(await page.locator('#new-lang').inputValue(),'fa');
 await page.locator('#new-description').fill('توضیح');
 await page.locator('#new-form button[type=submit]').click();await waitStatus('Local draft created');
 assert.equal(await page.locator('#source').inputValue(),importBody);
 const draftWrites=writeCount;
 await page.locator('#import-replace').click();
 await page.locator('#import-file').setInputFiles({name:'replace.md',mimeType:'text/markdown',buffer:Buffer.from('---\ntitle: "Ignored title"\npermalink: /wrong/\n---\n# replacement\n\nسلام\n')});
 await waitStatus('File imported into the existing article');
 assert.equal(await page.locator('#source').inputValue(),'# replacement\n\nسلام\n');
 assert.equal(await page.locator('#title').inputValue(),'فایل واردشده');
 assert.equal(await page.locator('#route').textContent(),'/fa/writing/imported/');
 assert.equal(writeCount,draftWrites);
 await page.locator('#import-replace').click();
 await page.locator('#import-file').setInputFiles({name:'bad.html',mimeType:'text/html',buffer:Buffer.from('<p>bad</p>')});
 await waitStatus('Replacement must match');
 assert.equal(await page.locator('#source').inputValue(),'# replacement\n\nسلام\n');
 // Full HTML import becomes an HTML post, not a Markdown conversion.
 await page.locator('#import-new').click();
 await page.locator('#import-file').setInputFiles({name:'example.html',mimeType:'text/html',buffer:Buffer.from('<!doctype html><html><head><title>Ignored</title></head><body><h1>HTML title</h1><p>Hello</p></body></html>')});
 await page.locator('#new-heading').waitFor();
 await page.locator('#new-description').fill('Example');
 await page.locator('#new-form button[type=submit]').click();await waitStatus('Local draft created');
 assert.equal(await page.locator('#source').inputValue(),'<h1>HTML title</h1><p>Hello</p>');
 assert.match(await page.locator('#kind').textContent(),/HTML/);assert.equal(await page.locator('html').getAttribute('lang'),'en');assert.equal(await page.locator('.doc[data-path="'+path+'"]').count(),0);
 await page.locator('#submit').click();await waitStatus('Committed to GitHub');assert.equal(commitCount,3);
 assert([...main.keys()].some(x=>x.endsWith('-en-example.html')));
 // Cancel without writes, then confirm one-click deletion. Simulate a lost mutation response.
 await page.locator('.doc[data-path="'+en+'"]').click();await waitStatus('Loaded from GitHub');
 const beforeDelete=writeCount;confirmDelete=false;
 await page.locator('#delete-post').click();await page.waitForTimeout(100);
 assert.equal(writeCount,beforeDelete);assert(main.has(en));confirmDelete=true;loseResponse=true;
 await page.locator('#delete-post').click();await waitStatus('network request failed');assert.equal(commitCount,4);assert(!main.has(en));assert(main.has(path));assert.match(main.get('retired-posts/en-document-parsing.html').toString(),/data-retired-post/);
 const deletionWrites=writeCount;
 await page.locator('#submit').click();await page.locator('#empty').waitFor();
 await page.locator('.doc[data-path="'+en+'"]').waitFor({state:'detached'});assert.equal(writeCount,deletionWrites);assert.equal(commitCount,4);
 console.log('Verified Markdown/HTML imports, replacement URL preservation, cross-format rejection, cancel, atomic one-click deletion and lost-response recovery.');
 await page.locator('#locale-fa').click();await page.locator('.doc[data-path="'+path+'"]').waitFor();assert.equal(await page.locator('.doc[data-path="'+en+'"]').count(),0);
 // Date-only update of an already published article through the visible dashboard action.
 const beforeDate=main.get(path).toString('utf8'),dateCommits=commitCount,finalDay='2024-07-12';
 await page.locator('.dashboard-row[data-path="'+path+'"] .dash-edit').click();await waitStatus('Loaded from GitHub');
 assert(await page.locator('#post-date').isVisible());assert(await page.locator('#post-date').isEnabled());assert.equal(await page.locator('#metadata').getAttribute('open'),null);
 await page.locator('#post-date').fill(finalDay);await waitStatus('Date saved locally');
 await page.locator('#dashboard-home').click();assert.equal(await page.locator('.dashboard-row[data-path="'+path+'"] time').textContent(),finalDay);
 await page.locator('.dashboard-row[data-path="'+path+'"] .dash-edit').click();await waitStatus('Recovered local draft');assert.equal(await page.locator('#post-date').inputValue(),finalDay);
 await page.locator('#submit').click();await waitStatus('Committed to GitHub');assert.equal(commitCount,dateCommits+1);assert.equal(main.get(path).toString('utf8'),beforeDate.replace(/^date:.*$/m,'date: '+JSON.stringify(finalDay+' 00:00:00 +0000')));
 await page.locator('#dashboard-home').click();assert.equal(await page.locator('.dashboard-row[data-path="'+path+'"] time').textContent(),finalDay);
 console.log('Verified default dashboard, locale-specific counts, draft filtering, return navigation, always-visible date picker, and date-only publishing with unchanged article bytes.');
 await page.screenshot({path:out+'/publication-studio-desktop.png'});await page.setViewportSize({width:375,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:out+'/publication-studio-mobile.png',fullPage:true});
 await page.locator('#logout').click();await page.locator('#login').waitFor();assert.equal(await page.evaluate(()=>localStorage.getItem('studio-remember-v1')),null);await signIn();await page.locator('#workspace').waitFor();assert.equal(await page.evaluate(()=>localStorage.getItem('studio-remember-v1')),null);
 const target=[...main.keys()].find(x=>x.endsWith('-fa-studio-test-post.md'));const updated=main.get(target).toString().replace(/^title:.*$/m,'title: "عنوان تغییر یافته بیرون از استودیو"');main.set(target,Buffer.from(updated));await page.reload();await page.locator('#login').waitFor();await signIn();await page.locator('#workspace').waitFor();await page.locator('.doc[data-path="'+target+'"] span').filter({hasText:'عنوان تغییر یافته بیرون از استودیو'}).waitFor();
 await page.evaluate(()=>localStorage.setItem('studio-remember-v1',JSON.stringify({token:'github_'+'pat_studio_test',expires:Date.now()-1})));await page.reload();await page.locator('#login').waitFor();assert.equal(await page.evaluate(()=>localStorage.getItem('studio-remember-v1')),null);await page.evaluate(()=>localStorage.setItem('studio-remember-v1',JSON.stringify({token:'github_'+'pat_revoked_test',expires:Date.now()+60000})));await page.reload();await page.getByText('GitHub rejected the token. Sign out and check its expiration.').waitFor();assert.equal(await page.evaluate(()=>localStorage.getItem('studio-remember-v1')),null);assert(await page.locator('#workspace').isHidden());assert.deepEqual(errors,[]);await ctx.close();console.log('Verified owner rejection, UTF-8/source integrity, local recovery, large preview, conflict prevention, ambiguous-response commit recovery, new post + atomic image commit, responsive layout.');
 }finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
