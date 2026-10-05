// Local unit test of generated page logic with a minimal in-memory DOM.
// This does not automate a browser or claim visual QA/human adjudication.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const input = fs.readFileSync(process.argv[2], 'utf8');
const source = input.startsWith('<!doctype html>') ? input.split('<script>')[1].split('</script>')[0] : input;
const elements = new Map();
let exported = null;
let alerts = 0;
function element(tag, id='') {
  return {tag,id,children:[],value:'',hidden:false,disabled:false,textContent:'',
    append(...nodes){this.children.push(...nodes); if(this.tag==='select'&&!this.value&&nodes[0])this.value=nodes[0].value;},
    replaceChildren(...nodes){this.children=[...nodes];}, click(){if(this.onclick)this.onclick();}};
}
for(const id of ['story','download','title','prefix','source','interpretation','reveal','graphs','judgment','decision','next'])
  elements.set(id,element(id==='story'?'select':'div',id));
const context = vm.createContext({
  document:{getElementById:id=>elements.get(id),createElement:tag=>element(tag)},
  alert:()=>alerts++, Blob,
  URL:{createObjectURL:blob=>{exported=blob;return 'blob:local-test';},revokeObjectURL:()=>{}},
  Date,
});
vm.runInContext(source,context,{timeout:5000});
const get=id=>elements.get(id);
const firstTitle=get('title').textContent;
const firstSource=get('source').textContent;
assert(firstTitle.includes('story_'));
assert.equal(get('graphs').hidden,true);
assert.equal(get('graphs').children.length,0);
get('reveal').onclick();
assert.equal(alerts,1); assert.equal(get('graphs').hidden,true);
get('interpretation').value='TEST ONLY: independent interpretation before reveal';
get('reveal').onclick();
assert.equal(get('interpretation').disabled,true);
assert.equal(get('graphs').hidden,false); assert.equal(get('graphs').children.length,3);
const firstStory=get('story').value;
const secondStory=get('story').children[1].value;
get('story').value=secondStory; get('story').onchange();
assert.equal(get('graphs').hidden,true);
get('story').value=firstStory; get('story').onchange();
assert.equal(get('source').textContent,firstSource);
assert.equal(get('interpretation').disabled,true);
assert.equal(get('interpretation').value,'TEST ONLY: independent interpretation before reveal');
assert.equal(get('graphs').hidden,false);
get('next').onclick(); assert.equal(alerts,2); assert.equal(get('title').textContent,firstTitle);
get('decision').value='TEST ONLY: no human validation; exercise save and next';
get('next').onclick();
assert.notEqual(get('title').textContent,firstTitle);
assert.equal(get('graphs').hidden,true);
get('download').onclick();
exported.text().then(text=>{
  const report=JSON.parse(text); assert.equal(report.responses.length,1);
  assert.match(report.packet_sha256,/^[0-9a-f]{64}$/);
  assert.equal(report.responses[0].interpretation,'TEST ONLY: independent interpretation before reveal');
  assert.equal(report.responses[0].judgment,'TEST ONLY: no human validation; exercise save and next');
  console.log(JSON.stringify({passed:8,checks:['source displayed before graph','blank interpretation blocks reveal','reveal freezes interpretation','new story starts source-only','return preserves prior frozen interpretation','blank judgment blocks progression','saved judgment advances without revealing next graph','local export contains interpretation and judgment'],visual_qa:false,human_validation:false},null,2));
});
