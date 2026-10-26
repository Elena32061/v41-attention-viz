// Pure shape arithmetic; no measured traffic or latency is used here.
export const DEFAULT = {op:'linear', b:32, q:1, ctx:14, pool:'sparse', ratio:2, dtype:'bf16'};
export const OPS = {
 linear:{label:'普通线性层',short:'Q 低秩投影 · Q_A',family:'linear'},
 up:{label:'MoE · gate + up',short:'单个专家 · 融合 gate + up',family:'moe'},
 down:{label:'MoE · down',short:'单个专家 · down',family:'moe'},
 qk:{label:'Attention · QK',short:'单请求、单 head · QKᵀ',family:'attention'},
 pv:{label:'Attention · PV',short:'单请求、单 head · PV',family:'attention'},
 index:{label:'Indexer · 打分',short:'单请求、单 index head · 全扫描',family:'index'}
};
export function normalize(raw={}) {
 const integer=(x,d,lo,hi)=>{const n=Number(x);return Number.isFinite(n)?Math.max(lo,Math.min(hi,Math.round(n))):d;};
 const op=Object.hasOwn(OPS,raw.op)?raw.op:DEFAULT.op;
 const q=[1,6,8].includes(Number(raw.q))?Number(raw.q):DEFAULT.q;
 const dtype=['bf16','fp8','w4a8'].includes(raw.dtype)?raw.dtype:DEFAULT.dtype;
 return {op,b:integer(raw.b??DEFAULT.b,DEFAULT.b,1,512),q,ctx:integer(raw.ctx??DEFAULT.ctx,DEFAULT.ctx,8,19),pool:['dense','sparse','swa'].includes(raw.pool)?raw.pool:DEFAULT.pool,ratio:Number(raw.ratio)===1?1:2,dtype:dtype==='w4a8'&&OPS[op].family!=='moe'?'bf16':dtype};
}
export function shape(raw) {
 const s=normalize(raw), L=2**s.ctx, T=s.b*s.q, S=s.pool==='sparse'?Math.min(512,L):s.pool==='swa'?Math.min(128,L):L;
 const common={...s,L,S,family:OPS[s.op].family,label:OPS[s.op].short};
 if(s.op==='linear')return {...common,M:T,K:5120,N:1280,left:'输入 token',right:'投影权重',output:'Q 潜向量',meaning:['本地 token 行数 b × q','输出特征宽度','输入特征宽度'],changes:['随 b、q 变化','固定为 1280','固定为 5120'],instances:1,scope:'一个本地线性层 GEMM'};
 if(s.op==='up'||s.op==='down')return {...common,M:T,K:s.op==='up'?5120:2304,N:s.op==='up'?4608:5120,left:'专家输入',right:'该专家的权重',output:s.op==='up'?'gate 与 up 输出':'专家输出',meaning:['每个专家的 token 数 mₑ','专家输出宽度','专家输入宽度'],changes:['均匀路由期望 b × q','固定专家结构','固定专家结构'],instances:6,scope:'一个专家 GEMM；本卡有 6 个独立专家'};
 if(s.op==='qk')return {...common,M:s.q,K:512,N:S,left:'Query',right:'Keyᵀ',output:'逻辑 scores',meaning:['本请求 query 数 q','参与打分的 key 数 S','head 特征宽度'],changes:['随 q 变化，不乘 b','随有效 KV 数变化','固定为 512'],instances:s.b*64,scope:`单请求、单 head；共 ${s.b} 请求 × 64 heads`};
 if(s.op==='pv')return {...common,M:s.q,K:S,N:512,left:'逻辑概率 P',right:'Value',output:'Attention 输出',meaning:['本请求 query 数 q','输出 head 宽度','参与加权的 value 数 S'],changes:['随 q 变化，不乘 b','固定为 512','随有效 KV 数变化'],instances:s.b*64,scope:`单请求、单 head；共 ${s.b} 请求 × 64 heads`};
 return {...common,M:s.q,K:128,N:Math.floor(L/s.ratio),left:'Index Query',right:'Index Keyᵀ',output:'逻辑 index scores',meaning:['本请求 query 数 q','压缩历史长度 L / r','index head 宽度'],changes:['随 q 变化，不乘 b','随上下文增长','固定为 128'],instances:s.b*32,scope:`单请求、单 index head；共 ${s.b} 请求 × 32 heads`};
}
export function cost(p,M=p.M) {
 const sizes=p.dtype==='fp8'?[1,1,1]:p.dtype==='w4a8'?[1,.5,2]:[2,2,2];
 const elements=[M*p.K,p.K*p.N,M*p.N];
 const bytes=elements.map((v,i)=>v*sizes[i]);
 const flops=2*M*p.N*p.K,total=bytes.reduce((a,b)=>a+b,0);
 return {sizes,elements,bytes,flops,total,ai:flops/total,elementAI:flops/elements.reduce((a,b)=>a+b,0),ceiling:2*p.N*p.K/(sizes[0]*p.K+sizes[2]*p.N)};
}
export function contextNote(p) {
 if(p.family==='linear')return '上下文变长，不会把输入特征 5120 变大。Decode 只把当前 token 送进线性层；此处固定模型与 TP1。';
 if(p.family==='moe')return 'EP64 均匀路由：mₑ = 64b × q × 6 / 384 = bq。每卡 6 个专家各有自己的权重；6bq 是分配总数，不是每个专家的 M。实际路由可能偏斜。';
 if(p.family==='index')return '全扫描 indexer 随压缩后的历史长度增长：N = L / r，K = 128。这里不展示后续候选打分；32 个 index heads 单独记为批维。';
 return 'QK 中 S 在 N 轴，PV 中同一个 S 换到 K 轴。不同请求的 KV 不同，b 是额外批维；不能把 M 写成 bq 并假设共享一份 KV。';
}
export const TOY_A=[[1,2,1,3],[2,1,3,1],[1,3,2,2]];
export const TOY_B=[[2,1,3],[1,3,2],[3,2,1],[2,1,2]];
export function toyStep(step) {
 const t=Math.max(0,Math.min(36,Math.floor(step))),out=Array.from({length:3},()=>[0,0,0]);
 for(let z=0;z<t;z++){const cell=Math.floor(z/4),i=Math.floor(cell/3),j=cell%3,k=z%4;out[i][j]+=TOY_A[i][k]*TOY_B[k][j];}
 const z=Math.min(t,35),cell=Math.floor(z/4);
 return {i:Math.floor(cell/3),j:cell%3,k:z%4,out,done:t===36};
}
