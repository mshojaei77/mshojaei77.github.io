const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync('assets/js/analytics.js','utf8');
function run({protocol='https:',hostname='mshojaei77.github.io',pathname='/fa/writing/example/',site='ci-fixture-only',current=true}={}){
 const appended=[];
 const document={currentScript:current?{dataset:{productionHost:'mshojaei77.github.io',goatcounterSite:site}}:null,createElement:tag=>({tag,dataset:{}}),head:{appendChild:s=>appended.push(s)}};
 vm.runInNewContext(source,{document,location:{protocol,hostname,pathname}});
 return appended;
}
const [script]=run();
assert.equal(script.src,'https://gc.zgo.at/count.v5.js');
assert.equal(script.crossOrigin,'anonymous');
assert.equal(script.integrity,'sha384-atnOLvQb9t+jTSipvd75X2yginT4PjVbqDdlJAmxMm+wYElFmeR6EmLP5bYeoRVQ');
assert.equal(script.async,true);
assert.equal(script.dataset.goatcounter,'https://ci-fixture-only.goatcounter.com/count');
for(const settings of [{hostname:'localhost'},{hostname:'other.example'},{protocol:'http:'},{pathname:'/admin/'},{pathname:'/admin'},{site:''},{site:'foo.example.com'},{site:'a'.repeat(64)},{current:false}])assert.equal(run(settings).length,0,JSON.stringify(settings));
console.log('Verified pinned analytics loader, production host and admin isolation without network calls.');
