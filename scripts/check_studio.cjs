const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {createHash}=require('node:crypto');
const origin=process.env.SITE_ORIGIN||'http://127.0.0.1:8765';
const out='/tmp/publication-screenshots';
const path='_posts/2026-11-26-document-parsing.md';
const en='_posts/2026-11-26-document-parsing-en.html';
const repo='/repos/mshojaei77/mshojaei77.github.io';
const source=fs.readFileSync(path,'utf8');
const sha=b=>createHash('sha1').update('blob '+b.length+'\0').update(b).digest('hex');
const entry=(p,b)=>({path:p,name:p.split('/').pop(),type:'file',encoding:'base64',size:b.length,sha:sha(b),content:b.toString('base64')});
const main=new Map([[path,Buffer.from(source)],[en,fs.readFileSync(en)]]);
const trees=new Map(),commits=new Map(),branches=new Map(),blobs=new Map(),prs=new Map();
let denied=0,writeCount=0,prCreates=0,loseResponse=true,mainChanged=false,treeSeq=0,commitSeq=0,prSeq=0;
const mainRef='a'.repeat(40);
const headers={'access-control-allow-origin':'*','access-control-allow-methods':'GET,POST,OPTIONS','access-control-allow-headers':'authorization,content-type,x-github-api-version,accept','content-type':'application/json'};
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
(async()=>{const browser=await chromium.launch({headless:true});try{
 const ctx=await browser.newContext({viewport:{width:1440,height:900},acceptDownloads:true});
 const page=await ctx.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('dialog',x=>x.accept());
 await ctx.route('https://api.github.com/**',async route=>{
  const req=route.request(),u=new URL(req.url()),p=decodeURIComponent(u.pathname),method=req.method(),token=req.headers().authorization;
  const reply=(status,json)=>route.fulfill({status,headers,body:JSON.stringify(json)});
  if(method==='OPTIONS')return route.fulfill({status:204,headers});
  if(p==='/user'){if(token==='Bearer outsider'){denied++;return reply(200,{id:99,login:'other'});}if(token!=='Bearer test-owner')return reply(401,{});return reply(200,{id:76538971,login:'mshojaei77'});}
  if(token!=='Bearer test-owner')return reply(401,{});
  if(p===repo)return reply(200,{full_name:'mshojaei77/mshojaei77.github.io',owner:{id:76538971},permissions:{push:true}});
  if(p===repo+'/contents/_posts')return reply(200,[...main].map(([p,b])=>({path:p,type:'file',sha:sha(b)})));
  if(p.startsWith(repo+'/contents/')){const name=p.slice((repo+'/contents/').length),ref=u.searchParams.get('ref'),files=ref==='main'||ref===mainRef?main:commits.get(ref);let b=files?.get(name);if(mainChanged&&name===path&&(ref==='main'||ref===mainRef))b=Buffer.from(source+'\nchanged externally');return b?reply(200,entry(name,b)):reply(404,{});}
  if(p.startsWith(repo+'/git/ref/heads/')){const name=p.slice((repo+'/git/ref/heads/').length);const c=name==='main'?mainRef:branches.get(name);return c?reply(200,{object:{sha:c}}):reply(404,{});}
  if(p===repo+'/git/commits/'+mainRef)return reply(200,{tree:{sha:'base-tree'}});
  if(p===repo+'/pulls'&&method==='GET'){const name=u.searchParams.get('head').split(':').slice(1).join(':');return reply(200,prs.has(name)?[prs.get(name)]:[]);}
  if(method!=='POST')return reply(404,{});
  writeCount++;const body=req.postDataJSON();
  if(p===repo+'/git/blobs'){const b=Buffer.from(body.content,'base64'),id=sha(b);blobs.set(id,b);return reply(201,{sha:id});}
  if(p===repo+'/git/trees'){assert.equal(body.base_tree,'base-tree');const files=new Map(main);for(const x of body.tree){assert(x.path.startsWith('_posts/')||x.path.startsWith('assets/uploads/'));files.set(x.path,x.content===undefined?blobs.get(x.sha):Buffer.from(x.content));}const id='tree-'+(++treeSeq);trees.set(id,files);return reply(201,{sha:id});}
  if(p===repo+'/git/commits'){assert.deepEqual(body.parents,[mainRef]);const id='commit-'+(++commitSeq);commits.set(id,trees.get(body.tree));return reply(201,{sha:id});}
  if(p===repo+'/git/refs'){assert(body.ref.startsWith('refs/heads/studio/'));const name=body.ref.slice('refs/heads/'.length);assert(!branches.has(name));branches.set(name,body.sha);return reply(201,{object:{sha:body.sha}});}
  if(p===repo+'/pulls'){assert.equal(body.base,'main');assert(branches.has(body.head));assert(!prs.has(body.head));prCreates++;const pr={number:++prSeq,state:'open'};prs.set(body.head,pr);if(loseResponse){loseResponse=false;return route.abort('failed');}return reply(201,pr);}
  throw Error('Unexpected mutation '+p);
 });
 async function signIn(token='test-owner'){await page.locator('#token').fill(token);await page.getByRole('button',{name:'Sign in',exact:true}).click();}
 const waitStatus=async re=>{await page.waitForFunction(s=>new RegExp(s,'i').test(document.querySelector('#status').textContent),re);};
 assert((await page.goto(origin+'/admin/')).ok());await page.locator('#login').waitFor();
 await signIn('outsider');await page.getByText('Access denied: this workspace is reserved for its owner.').waitFor();assert.equal(denied,1);assert.equal(writeCount,0);assert(await page.locator('#workspace').isHidden());
 await signIn();await page.locator('#workspace').waitFor();assert.equal(await page.locator('#token').inputValue(),'');
 const openTime=Date.now();await page.locator('.doc[data-path="'+path+'"]').click();await page.locator('#editor-pane').waitFor();await waitStatus('Loaded from GitHub');
 const body=source.replace(/^(?:\uFEFF)?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/,'');assert.equal(await page.locator('#source').inputValue(),body);assert.equal(await page.locator('#source').getAttribute('dir'),'rtl');
 assert(await page.locator('#preview').isHidden());
 const times=await page.locator('#source').evaluate(el=>{const times=[];const original=el.value;for(let i=0;i<20;i++){const t=performance.now();el.setRangeText('x',el.value.length,el.value.length,'end');el.dispatchEvent(new Event('input'));times.push(performance.now()-t);}el.value=original;el.dispatchEvent(new Event('input'));return times.sort((a,b)=>a-b);});
 const p95=times[Math.ceil(times.length*.95)-1];assert(p95<100,'Editor input p95 '+p95+' ms');console.log('Studio actual manuscript bytes='+Buffer.byteLength(source)+' mocked open ms='+(Date.now()-openTime)+' synchronous input event p95 ms='+p95.toFixed(2));
 await page.getByRole('button',{name:'Preview',exact:true}).click();await page.frameLocator('#preview').locator('svg').first().waitFor();assert.equal(await page.frameLocator('#preview').locator('svg').count(),25);
 await page.getByRole('button',{name:'Preview',exact:true}).click();const marker='\n<!-- Studio test edit -->\n';await page.locator('#source').evaluate((el,m)=>{el.setRangeText(m,el.value.length,el.value.length,'end');el.dispatchEvent(new Event('input'));},marker);
 await page.getByRole('button',{name:'Save locally'}).click();await waitStatus('Draft saved on this device');assert.equal(writeCount,0);
 await page.reload();await page.locator('#login').waitFor();await signIn();await page.locator('.doc[data-path="'+path+'"]').click();await waitStatus('Recovered local draft');assert.equal(await page.locator('#source').inputValue(),body+marker);
 mainChanged=true;await page.getByRole('button',{name:'Submit PR'}).click();await waitStatus('Main has changed');assert.equal(writeCount,0);mainChanged=false;
 await page.getByRole('button',{name:'Submit PR'}).click();await waitStatus('network request failed');assert.equal(prCreates,1);const after=writeCount;
 await page.getByRole('button',{name:'Submit PR'}).click();await waitStatus('Pull request #1 created');assert.equal(prCreates,1);assert.equal(writeCount,after);
 const submitted=commits.get([...branches.values()][0]).get(path).toString('utf8');assert.equal(submitted,source+marker);assert(await page.locator('#source').evaluate(e=>e.readOnly));
 assert.equal(await page.locator('#pr').getAttribute('href'),'https://github.com/mshojaei77/mshojaei77.github.io/pull/1');
 await page.getByRole('button',{name:'+ New',exact:true}).click();await page.locator('#new-title').fill('آزمایش جدید');await page.locator('#new-slug').fill('studio-test-post');await page.locator('#new-description').fill('توضیح آزمایشی');await page.getByRole('button',{name:'Create draft'}).click();await waitStatus('Local draft created');assert.equal(writeCount,after);
 await page.locator('#source').fill('## Hello\n\n**سلام**');const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aawAAAABJRU5ErkJggg==','base64');await page.locator('#media').setInputFiles({name:'pixel.png',mimeType:'image/png',buffer:png});await waitStatus('Image staged locally');
 await page.getByRole('button',{name:'Submit PR'}).click();await waitStatus('Pull request #2 created');assert.equal(prCreates,2);const newest=commits.get([...branches.values()].at(-1));const newPath=[...newest.keys()].find(x=>x.includes('studio-test-post'));const asset=[...newest.keys()].find(x=>x.startsWith('assets/uploads/'));assert(newPath&&asset);assert(newest.get(newPath).toString().includes('/fa/writing/studio-test-post/'));assert(newest.get(newPath).toString().includes('/'+asset));assert.deepEqual(newest.get(asset),png);
 await page.screenshot({path:out+'/publication-studio-desktop.png'});await page.setViewportSize({width:375,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:out+'/publication-studio-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);await ctx.close();console.log('Verified owner rejection, UTF-8/source integrity, local recovery, large preview, conflict prevention, ambiguous-response PR recovery, new post + atomic image commit, responsive layout.');
 }finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
