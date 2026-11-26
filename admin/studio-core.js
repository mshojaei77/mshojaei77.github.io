export const OWNER = 'mshojaei77';
export const OWNER_ID = 76538971;
export const REPO = OWNER + '/mshojaei77.github.io';
export const ROOT = '/repos/' + REPO;
export const SITE = 'https://mshojaei77.github.io';
export const MAX_BYTES = 950000;
export const PAGES = {'about.md':'About · EN','fa/about/index.html':'دربارهٔ من · FA','roadmap.md':'Roadmap · EN','fa/roadmap/index.md':'نقشهٔ راه · FA'};
export const LABELS = {'_posts/2026-11-26-document-parsing.md':'مهندسی Document Parsing · FA','_posts/2026-11-26-document-parsing-en.html':'Document Parsing tutorial · EN',"_posts/2026-11-26-fa-market-analysis.md":"تحلیل بازار مهندسی LLM · FA","_posts/2026-11-26-en-market-analysis.md":"LLM Engineering Market Analysis · EN","_posts/2026-11-26-fa-vector-databases.html":"مقایسهٔ پایگاه‌های دادهٔ برداری · FA","_posts/2026-11-26-en-vector-databases.html":"Vector Database Comparison · EN","_posts/2026-11-26-en-open-source-ai.html":"Open Source AI: Driving Freedom and Shaping the Future","_posts/2026-11-26-fa-open-source-ai.html":"هوش مصنوعی متن‌باز؛ آزادی و آینده","_posts/2026-11-26-en-resources.md":"Learning Resources","_posts/2026-11-26-fa-resources.md":"منابع یادگیری","_posts/2026-11-26-en-interview.md":"LLM Interview Questions","_posts/2026-11-26-fa-interview.md":"سؤالات مصاحبهٔ LLM","_posts/2026-11-26-en-mini-roadmap.html":"LLM Development Roadmap — Compact","_posts/2026-11-26-fa-mini-roadmap.html":"نقشهٔ راه فشردهٔ LLM","_posts/2026-11-26-en-data-preparation.md":"Data Preparation for LLMs: Cleaning Guide","_posts/2026-11-26-fa-data-preparation.md":"آماده‌سازی داده برای مدل‌های زبانی","_posts/2026-11-26-en-embeddings.md":"Embeddings Tutorial: Word Vectors for LLMs","_posts/2026-11-26-fa-embeddings.md":"آموزش Embedding؛ بردارهای واژه در LLM","_posts/2026-11-26-en-intro-to-llms.md":"Intro to LLMs: Large Language Models Basics","_posts/2026-11-26-fa-intro-to-llms.md":"آشنایی با مدل‌های زبانی بزرگ (LLM)","_posts/2026-11-26-en-neural-networks.md":"Neural Networks Tutorial: Foundations for LLMs","_posts/2026-11-26-fa-neural-networks.md":"آموزش شبکه‌های عصبی؛ مبانی LLM","_posts/2026-11-26-en-tokenization.md":"Tokenization Tutorial: Text to Tokens for LLMs","_posts/2026-11-26-fa-tokenization.md":"آموزش Tokenization؛ تبدیل متن به توکن","_posts/2026-11-26-en-traditional-language-models.md":"Traditional Language Models Tutorial: Pre-Transformers Era","_posts/2026-11-26-fa-traditional-language-models.md":"مدل‌های زبانی سنتی؛ پیش از Transformer","_posts/2026-11-26-en-transformers.md":"Transformers Tutorial: Attention Revolution for LLMs","_posts/2026-11-26-fa-transformers.md":"آموزش Transformer؛ سازوکار Attention",...PAGES};
export const allowed = p => typeof p === 'string' && (Object.hasOwn(PAGES,p) || /^_posts\/\d{4}-\d{2}-\d{2}-[A-Za-z0-9_-]+\.(md|markdown|html)$/.test(p));
export function documentLanguage(path){
 const old={'_posts/2026-11-26-document-parsing.md':'fa','_posts/2026-11-26-document-parsing-en.html':'en','about.md':'en','roadmap.md':'en','fa/about/index.html':'fa','fa/roadmap/index.md':'fa'};
 return old[path]||(/^_posts\/\d{4}-\d{2}-\d{2}-(fa|en)-[A-Za-z0-9_-]+\.(md|markdown|html)$/.exec(path)?.[1]??null);
}
export function postRoute(path){
 const lang=documentLanguage(path);
 if(!path.startsWith('_posts/')||!lang)return null;
 const old={'_posts/2026-11-26-document-parsing.md':'document-parsing','_posts/2026-11-26-document-parsing-en.html':'document-parsing'};
 const slug=old[path]||/^\d{4}-\d{2}-\d{2}-(?:fa|en)-([a-z0-9]+(?:-[a-z0-9]+)*)\.(?:md|markdown|html)$/.exec(path.split('/').pop())?.[1];
 return slug?(lang==='fa'?'/fa':'')+'/writing/'+slug+'/':null;
}
export const encodedPath = p => p.split('/').map(encodeURIComponent).join('/');
export const bytes = s => new TextEncoder().encode(s);
export function to64(a){let s='';for(let i=0;i<a.length;i+=16384)s+=String.fromCharCode(...a.subarray(i,i+16384));return btoa(s);}
export function from64(s){return Uint8Array.from(atob(s.replace(/\s/g,'')),c=>c.charCodeAt(0));}
export const decode = s => new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(from64(s));
export async function blobSha(a){const prefix=bytes('blob '+a.length+'\0');const all=new Uint8Array(prefix.length+a.length);all.set(prefix);all.set(a,prefix.length);return [...new Uint8Array(await crypto.subtle.digest('SHA-1',all))].map(x=>x.toString(16).padStart(2,'0')).join('');}
export function splitSource(s){const m=/^(?:\uFEFF)?---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/.exec(s);if(!m)throw Error('Missing Jekyll front matter; source was not modified.');return {header:m[0],body:s.slice(m[0].length)};}
export function scalar(h,k){const m=new RegExp('^'+k+':[ \\t]*(.*)$','m').exec(h);if(!m)return '';let v=m[1].trim();if(v.startsWith('"')){try{return JSON.parse(v);}catch{throw Error('Unsupported quoted '+k+'; use GitHub source editor.');}}if(v.startsWith("'")&&v.endsWith("'"))return v.slice(1,-1).replace(/''/g,"'");if(/^[>|]/.test(v))throw Error('Multiline '+k+' requires source editing in GitHub.');return v;}
export function setScalar(h,k,v){if(scalar(h,k)===v)return h;const re=new RegExp('^'+k+':[ \\t]*[^\\r\\n]*','m');const line=k+': '+JSON.stringify(v);if(re.test(h))return h.replace(re,()=>line);return h.replace(/(\r?\n)---(\r?\n|$)$/,(_,a,b)=>a+line+a+'---'+b);}
export const meta=s=>{const {header}=splitSource(s);return Object.fromEntries(['title','description','excerpt','permalink','lang','date','layout'].map(k=>[k,scalar(header,k)]));};
export function bodyWithLineEndings(body,original){return original.includes('\r\n')&&!original.replace(/\r\n/g,'').includes('\n')?body.replace(/\r?\n/g,'\r\n'):body;}
export function createPost({title,description,slug,lang,date=new Date().toISOString().slice(0,10),extension='md',body=null}){if(!['fa','en'].includes(lang)||!title.trim()||!description.trim()||!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)||slug.length>100||['document-parsing','document-parsing-engineering'].includes(slug))throw Error('Choose a unique, non-reserved ASCII slug and fill in the required fields.');if(!['md','html'].includes(extension))throw Error('Unsupported article format.');const path='_posts/'+date+'-'+lang+'-'+slug+'.'+extension;const route=(lang==='fa'?'/fa':'')+'/writing/'+slug+'/';return {path,source:['---','layout: article','lang: '+lang,'article_kind: essay','show_article_header: '+!(body&&hasHeading(body,extension)),'title: '+JSON.stringify(title.trim()),'description: '+JSON.stringify(description.trim()),'excerpt: '+JSON.stringify(description.trim()),'date: '+date+' 00:00:00 +0000','permalink: '+route,'nav_exclude: true','image: /assets/img/social-card.png','feed:','  excerpt_only: true','tags: []','---','',''].join('\n').replace(/\n\n$/,body===null?'\n\n':'\n')+(body??''),baseSha:null,assets:[],attempt:null,pr:null};}

export function hasHeading(body,extension){return extension==='html'?/<h1(?:\s|>)/i.test(body):/^(?:#\s+|<h1(?:\s|>))/m.test(body);}
export function parseImport(name,buffer){
 const ext=name.toLowerCase().split('.').pop();if(!['md','markdown','html'].includes(ext))throw Error('Select a .md, .markdown or .html file.');
 if(buffer.byteLength>MAX_BYTES)throw Error('File exceeds the 950 KB source limit.');
 let source;try{source=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(buffer);}catch{throw Error('Import requires a valid UTF-8 file.');}
 if(source.includes('\0'))throw Error('Binary files are not supported.');
 let body=source,fields={};if(/^(?:\uFEFF)?---\r?\n/.test(source)){const p=splitSource(source);body=p.body;fields=meta(source);}
 const extension=ext==='html'?'html':'md';
 if(extension==='html'&&/^\s*(?:\uFEFF)?(?:<!doctype\s+html\b|<html\b)/i.test(body)){const m=/<body\b[^>]*>([\s\S]*)<\/body\s*>/i.exec(body);if(!m)throw Error('Full HTML documents need an explicit body element.');body=m[1];}
 if(!body.trim())throw Error('File has no article content.');
 const heading=extension==='html'?/<h1\b[^>]*>([\s\S]*?)<\/h1\s*>/i.exec(body):/^#\s+(.+)$/m.exec(body);
 const title=fields.title||heading?.[1]?.replace(/<[^>]*>/g,'').trim()||name.replace(/\.[^.]+$/,'');
 const lang=['fa','en'].includes(fields.lang)?fields.lang:/[\u0600-\u06ff]/.test(title+body.slice(0,4000))?'fa':'en';
 const slug=(fields.permalink?.match(/\/([a-z0-9]+(?:-[a-z0-9]+)*)\/$/)?.[1]||name.replace(/\.[^.]+$/,'').replace(/^\d{4}-\d{2}-\d{2}-/,'').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'')).slice(0,100).replace(/-$/,'');
 return {extension,body,title,lang,slug,description:fields.description||'',name};
}
export function retirement(path,source){
 if(!allowed(path)||!path.startsWith('_posts/'))throw Error('Only published posts can be deleted.');
 const m=meta(source);const match=/^\/(fa\/)?writing\/([a-z0-9]+(?:-[a-z0-9]+)*)\/$/.exec(m.permalink);
 if(!match||m.lang!==(match[1]?'fa':'en'))throw Error('Unsupported permalink; remove this post through GitHub.');
 const fa=m.lang==='fa',archive=fa?'/fa/writing/':'/writing/';
 const content=['---','layout: publication','lang: '+m.lang,'title: '+JSON.stringify(fa?'این نوشته حذف شده است':'Article removed'),'description: '+JSON.stringify(fa?'این نوشته دیگر منتشر نمی‌شود.':'This article is no longer published.'),'permalink: '+m.permalink,'sitemap: false','nav_exclude: true','retired_post: true','---','<section data-retired-post><h1>'+(fa?'این نوشته حذف شده است':'Article removed')+'</h1><p><a href="'+archive+'">'+(fa?'بازگشت به نوشته‌ها':'Back to writing')+'</a></p></section>',''].join('\n');
 return {path:'retired-posts/'+m.lang+'-'+match[2]+'.html',content};
}

export class ApiError extends Error{constructor(status,message){super(message);this.status=status;}}
export class GitHub{
 #token;
 constructor(token){this.#token=token;}
 clear(){this.#token='';}
 async call(path,{method='GET',body,missing=false}={}){if(!this.#token)throw Error('Sign in first.');let res;try{res=await fetch('https://api.github.com'+path,{method,headers:{Accept:'application/vnd.github+json',Authorization:'Bearer '+this.#token,'X-GitHub-Api-Version':'2022-11-28',...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined,cache:'no-store',redirect:'error',signal:AbortSignal.timeout(25000)});}catch{throw Error('GitHub network request failed or timed out. Your local draft is safe; retry when connected.');}if(missing&&res.status===404)return null;if(!res.ok){const code=res.status;throw new ApiError(code,code===401?'GitHub rejected the token. Sign out and check its expiration.':code===403?'GitHub denied access or rate limited this request. Check token permissions and try again.':code===409||code===422?'GitHub reported a conflict or rejected the requested change. Check the draft and branch before retrying.':'GitHub request failed (HTTP '+code+').');}return res.json();}
 async authenticate(){const u=await this.call('/user');if(u.id!==OWNER_ID||u.login!==OWNER)throw Error('Access denied: this workspace is reserved for its owner.');const r=await this.call(ROOT);if(r.full_name!==REPO||r.owner?.id!==OWNER_ID||r.permissions?.push!==true)throw Error('The token does not have confirmed write access to this repository.');return u;}
 async list(ref='main'){const a=await this.call(ROOT+'/contents/_posts?ref='+encodeURIComponent(ref));if(!Array.isArray(a))throw Error('Could not list posts.');if(a.length>=1000)throw Error('Post directory listing limit reached; use GitHub source editor.');return a.filter(x=>x.type==='file'&&allowed(x.path)).map(x=>({path:x.path,sha:x.sha})).concat(Object.keys(PAGES).map(path=>({path})));}
 async file(path,ref='main'){if(!allowed(path)&&!/^retired-posts\/(fa|en)-[a-z0-9-]+\.html$/.test(path)&&!/^assets\/uploads\/[a-zA-Z0-9_.-]+$/.test(path))throw Error('File path is outside Studio content.');return this.call(ROOT+'/contents/'+encodedPath(path)+'?ref='+encodeURIComponent(ref),{missing:true});}
 async read(path,ref='main'){const x=await this.file(path,ref);if(!x)return null;if(x.type!=='file'||x.encoding!=='base64'||x.size>MAX_BYTES)throw Error('This source is too large or not a UTF-8 text file.');const source=decode(x.content);if(await blobSha(bytes(source))!==x.sha)throw Error('Source integrity check failed.');return {source,sha:x.sha};}
 async ref(branch='main'){return this.call(ROOT+'/git/ref/heads/'+encodedPath(branch),{missing:true});}
 async pulls(branch){return this.call(ROOT+'/pulls?state=all&head='+encodeURIComponent(OWNER+':'+branch)+'&base=main');}
 async pr(number){if(!Number.isInteger(number)||number<1)throw Error('Invalid PR.');return this.call(ROOT+'/pulls/'+number);}
 async prepare(d){if(!allowed(d.path))throw Error('Invalid content path.');if(bytes(d.source).length>MAX_BYTES)throw Error('Article exceeds the 950 KB Studio source limit. Download the source and use GitHub instead.');const main=await this.ref();if(!main?.object?.sha)throw Error('Could not resolve main.');const base=main.object.sha;const current=await this.read(d.path,base);if((current?.sha??null)!==d.baseSha)throw Error('Main has changed since this draft was opened. Download your draft and reconcile with the current source; nothing was overwritten.');if(d.deleting){if(!current||!d.baseSha)throw Error('Post is not published.');const retired=retirement(d.path,current.source);if(await this.file(retired.path,base))throw Error('A retirement page already exists.');return {base,files:await this.submissionFiles(d)};}const m=meta(d.source);if(documentLanguage(d.path)!==m.lang||!m.title.trim()||!m.description.trim()||!m.permalink||!['fa','en'].includes(m.lang))throw Error('Missing required article metadata.');if(current){const old=meta(current.source);for(const k of ['permalink','lang','date','layout'])if(m[k]!==old[k])throw Error('Protected metadata changed: '+k);}else{if(!d.path.startsWith('_posts/')||!new RegExp('^'+(m.lang==='fa'?'/fa':'')+'/writing/[a-z0-9]+(?:-[a-z0-9]+)*/$').test(m.permalink))throw Error('Invalid new post permalink.');const retired=retirement(d.path,d.source);if(await this.file(retired.path,base))throw Error('This permalink belongs to a removed article. Choose another slug.');if(postRoute(d.path)!==m.permalink)throw Error('Filename does not match the canonical permalink.');const items=await this.list(base);if(items.some(item=>postRoute(item.path)===m.permalink))throw Error('This permalink is already published.');}
 if(current?.source===d.source&&!d.assets?.length)throw Error('There are no changes to publish.');return {base,files:await this.submissionFiles(d)};}
 async submissionFiles(d){if(!allowed(d.path))throw Error('Invalid content path.');if(d.deleting){if(!d.baseSha)throw Error('Only published posts can be deleted.');const retired=retirement(d.path,d.source);return [{path:d.path,sha:null,deleted:true},{...retired,sha:await blobSha(bytes(retired.content))}];}const files=[{path:d.path,sha:await blobSha(bytes(d.source)),content:d.source}];for(const asset of d.assets||[]){if(!/^assets\/uploads\/[a-f0-9-]+\.(png|jpg|webp|gif)$/.test(asset.path)||await blobSha(from64(asset.base64))!==asset.sha)throw Error('Invalid staged image.');files.push({path:asset.path,sha:asset.sha,base64:asset.base64});}return files;}

 // One GitHub commit with an explicit compare-and-swap on main; never force-push.
 async publish(d,saveAttempt){
  if(d.pr||d.attempt)throw Error('Resolve the previous pull request first.');
  const files=await this.submissionFiles(d);
  const fingerprint=files.map(f=>f.path+':'+f.sha).join('|');
  if(d.directAttempt){
   if(d.directAttempt.fingerprint!==fingerprint)throw Error('An earlier publish may have succeeded. Check its result before modifying this draft.');
   const head=await this.ref();
   const matches=await Promise.all(files.map(async f=>((await this.file(f.path,head.object.sha))?.sha??null)===f.sha));
   if(matches.every(Boolean))return {oid:head.object.sha,recovered:true,sourceSha:files[0].sha};
  }
  const {base}=await this.prepare(d);
  d.directAttempt={fingerprint,base};await saveAttempt();
  const fileChanges={additions:files.filter(f=>!f.deleted).map(f=>({path:f.path,contents:f.base64||to64(bytes(f.content))})),deletions:files.filter(f=>f.deleted).map(f=>({path:f.path}))};
  const query='mutation Publish($input: CreateCommitOnBranchInput!) { createCommitOnBranch(input: $input) { commit { oid url } } }';
  const result=await this.call('/graphql',{method:'POST',body:{query,variables:{input:{branch:{repositoryNameWithOwner:REPO,branchName:'main'},expectedHeadOid:base,message:{headline:(d.deleting?'Studio: remove ':'Studio: publish ')+meta(d.source).title.slice(0,100)},fileChanges}}}});
  if(result.errors?.length||!result.data?.createCommitOnBranch?.commit?.oid)throw Error('GitHub did not confirm the commit: '+(result.errors?.map(e=>e.message).join('; ')||'unknown response')+'. Retry to check the outcome; local draft retained.');
  return {...result.data.createCommitOnBranch.commit,sourceSha:files[0].sha};
 }
 async resolveLegacy(d){
  if(!d.attempt&&!d.pr)return;
  const pr=d.pr?await this.pr(d.pr.number):(await this.pulls(d.attempt.branch))[0];
  if(pr&&pr.state==='open'){d.pr={number:pr.number};return;}
  if(pr&&(pr.merged||pr.merged_at)){
   const current=await this.read(d.path);
   if(d.deleting&&!current){return 'deleted';}
   if(!current||current.source!==d.source)throw Error('The previous merged PR differs from your local draft. Download it and reconcile before publishing.');
   d.baseSha=current.sha;d.assets=[];
  }else if(!pr&&d.attempt&&await this.ref(d.attempt.branch)){
   // Legacy branch without a PR cannot have changed main; retain its Git history.
  }
  d.pr=null;d.attempt=null;d.deleting=false;
 }

}
let dbPromise;
function db(){if(!dbPromise)dbPromise=new Promise((resolve,reject)=>{const r=indexedDB.open('publication-studio-v1',1);r.onupgradeneeded=()=>r.result.createObjectStore('drafts',{keyPath:'path'});r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);r.onblocked=()=>reject(Error('Draft database is blocked by another tab.'));});return dbPromise;}
async function transaction(mode,action){const database=await db();return new Promise((resolve,reject)=>{const tx=database.transaction('drafts',mode);const req=action(tx.objectStore('drafts'));let result;req.onsuccess=()=>{result=req.result;};tx.oncomplete=()=>resolve(result);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error||Error('Local draft transaction aborted'));});}
export const drafts={all:()=>transaction('readonly',s=>s.getAll()),put:d=>transaction('readwrite',s=>s.put(d)),remove:p=>transaction('readwrite',s=>s.delete(p))};
