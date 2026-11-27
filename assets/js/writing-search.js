(()=>{'use strict';
const link=document.getElementById('site-search'), input=document.getElementById('writing-search'),kindSelect=document.getElementById('writing-kind'),filters=document.getElementById('writing-filters');
const normalize=s=>String(s||'').normalize('NFKC').toLowerCase().replace(/[يى]/g,'ی').replace(/ك/g,'ک').replace(/[\u064B-\u065F\u0670]/g,'').replace(/[\u200c\u200d]/g,' ').replace(/\s+/g,' ').trim();
if(input&&filters){
 const params=new URLSearchParams(location.search),buttons=[...filters.querySelectorAll('[data-topic]')],rows=[...document.querySelectorAll('.writing-list [data-post-topics]')],count=document.getElementById('writing-count'),fa=document.documentElement.lang==='fa';
 const records=rows.map(el=>({el,topics:el.dataset.postTopics.split(' '),kind:el.dataset.articleKind||'guide',text:normalize(el.textContent+' '+el.dataset.postTopics)}));
 let topic=buttons.some(b=>b.dataset.topic===params.get('topic'))?params.get('topic'):'';
 const kinds=[...kindSelect.options].map(o=>o.value);kindSelect.value=kinds.includes(params.get('kind'))?params.get('kind'):'';
 input.value=params.get('q')||'';filters.hidden=false;
 function update(save){const terms=normalize(input.value).split(' ').filter(Boolean),kind=kindSelect.value;let n=0;
  for(const r of records){const visible=(!topic||r.topics.includes(topic))&&(!kind||r.kind===kind)&&terms.every(t=>r.text.includes(t));r.el.hidden=!visible;if(visible)n++;}
  for(const b of buttons)b.setAttribute('aria-pressed',String(b.dataset.topic===topic));
  count.textContent=fa?(n.toLocaleString('fa-IR')+' نوشته'):(n+' '+(n===1?'article':'articles'));
  if(save){const u=new URL(location.href);for(const [k,v] of [['topic',topic],['kind',kind],['q',input.value.trim()]]){if(v)u.searchParams.set(k,v);else u.searchParams.delete(k);}u.searchParams.delete('search');history.replaceState(null,'',u);}
 }
 input.addEventListener('input',()=>update(true));kindSelect.addEventListener('change',()=>update(true));for(const b of buttons)b.addEventListener('click',()=>{topic=b.dataset.topic;update(true);});
 update(false);if(params.has('search'))input.focus();
 if(link)link.addEventListener('click',e=>{if(e.metaKey||e.ctrlKey||e.shiftKey||e.altKey)return;e.preventDefault();input.focus();});
}
document.addEventListener('keydown',e=>{if(e.defaultPrevented||e.isComposing||e.altKey||e.shiftKey||!(e.ctrlKey||e.metaKey)||e.key.toLowerCase()!=='k')return;if(e.target instanceof HTMLElement&&(e.target.isContentEditable||/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)))return;if(!link)return;e.preventDefault();if(input)input.focus();else location.assign(link.href);});
})();
