import assert from 'node:assert/strict';
import {shape,cost,normalize,toyStep,TOY_A,TOY_B} from './mnk-model.js';
const dims=p=>[p.M,p.N,p.K];
assert.deepEqual(dims(shape({op:'linear',b:32,q:1})),[32,1280,5120]);
assert.deepEqual(dims(shape({op:'up',b:32,q:6})),[192,4608,5120]);
assert.deepEqual(dims(shape({op:'down',b:32,q:6})),[192,5120,2304]);
for(const b of [1,32,512]){
 assert.deepEqual(dims(shape({op:'qk',b,q:6,ctx:14,pool:'dense'})),[6,16384,512]);
 assert.deepEqual(dims(shape({op:'pv',b,q:6,ctx:14,pool:'dense'})),[6,512,16384]);
 assert.deepEqual(dims(shape({op:'index',b,q:6,ctx:14,ratio:2})),[6,8192,128]);
}
assert.deepEqual(dims(shape({op:'qk',q:1,ctx:8,pool:'sparse'})),[1,256,512]);
assert.deepEqual(dims(shape({op:'qk',q:1,ctx:19,pool:'sparse'})),[1,512,512]);
assert.deepEqual(dims(shape({op:'pv',q:1,ctx:19,pool:'swa'})),[1,512,128]);
assert.equal(normalize({op:'qk',dtype:'w4a8'}).dtype,'bf16');
assert.equal(normalize({op:'up',dtype:'w4a8'}).dtype,'w4a8');
assert.equal(normalize({b:'NaN',ctx:100,q:5}).b,32);
assert.equal(normalize({b:'NaN',ctx:100,q:5}).ctx,19);
assert.equal(normalize({b:'NaN',ctx:100,q:5}).q,1);
const a=cost({M:3,K:4,N:3,dtype:'bf16'}),f=cost({M:3,K:4,N:3,dtype:'fp8'}),w=cost({M:3,K:4,N:3,dtype:'w4a8'});
assert.equal(a.flops,72);assert.deepEqual(a.bytes,[24,24,18]);assert.equal(a.ai,72/66);assert.equal(f.ai,a.ai*2);assert.deepEqual(w.bytes,[12,6,18]);
const p=shape({op:'up',dtype:'w4a8'});let previous=0;for(const m of [1,2,32,192,1024,8192]){const c=cost(p,m);assert.ok(c.ai>previous);assert.ok(c.ai<c.ceiling);previous=c.ai;}
const want=TOY_A.map(row=>TOY_B[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*TOY_B[k][j],0)));
assert.deepEqual(toyStep(36).out,want);assert.equal(toyStep(36).done,true);assert.deepEqual(toyStep(1).out,[[2,0,0],[0,0,0],[0,0,0]]);
console.log('MNK checks passed: layer/expert shapes, attention batch independence, sparse caps, dtype bytes, AI monotonicity, URL guards and all 36 multiply-accumulate steps.');
