import {DEFAULT,OPS,normalize,shape,cost,contextNote,TOY_A,TOY_B,toyStep} from './mnk-model.js';
const $=id=>document.getElementById(id),fmt=n=>n.toLocaleString('en-US'),COL={m:'#2569c9',n:'#b96a11',k:'#8255c6',ink:'#18253a',muted:'#627086',line:'#e2e7ef'};
let state=normalize({...DEFAULT,...Object.fromEntries(new URLSearchParams(location.search))});
let step=0,playing=false,timer=null,sweepTimer=null;
const el={op:$('op'),b:$('batch'),q:$('query'),ctx:$('context'),pool:$('pool'),ratio:$('ratio'),dtype:$('dtype')};
const compact=(n,base=1000)=>{const units=base===1024?['B','KiB','MiB','GiB']:['FLOPs','kFLOPs','MFLOPs','GFLOPs'];let i=0;while(n>=base&&i<3){n/=base;i++;}return [n.toLocaleString('en-US',{maximumFractionDigits:2}),units[i]];};
const text=(x,y,s,size=13,fill=COL.ink,anchor='start',weight=400)=>`<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" text-anchor="${anchor}" font-weight="${weight}">${s}</text>`;
function saveURL(){const params=new URLSearchParams(state);history.replaceState(null,'',`${location.pathname}?${params}`);}
function updateControls(){for(const [key,node] of Object.entries(el))node.value=state[key];const fam=OPS[state.op].family;$('poolControl').hidden=fam!=='attention';$('ratioControl').hidden=fam!=='index';$('dtype').querySelector('[value="w4a8"]').disabled=fam!=='moe';$('batchOut').textContent=fmt(state.b);$('contextOut').textContent=state.ctx>=10?`${2**state.ctx/1024}K`:`${2**state.ctx}`;}
function shapeDrawing(p){
 const dimLen=v=>Math.min(150,42+Math.log2(Math.max(1,v))*7),M=dimLen(p.M),N=dimLen(p.N),K=dimLen(p.K);
 function matrix(cx,w,h,top,left,name,shapeText,tColor,lColor){const x=cx-w/2,y=47;let out=`<rect class="shape-block" x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="#f5f7fb" stroke="#dce3ed"/>`;for(let a=1;a<5;a++)out+=`<path d="M ${x+w*a/5} ${y} v ${h} M ${x} ${y+h*a/5} h ${w}" stroke="#e3e8f0" fill="none"/>`;out+=text(cx,30,top,14,tColor,'middle',650)+text(x-12,y+h/2+5,left,14,lColor,'end',650)+text(cx,217,name,13,COL.ink,'middle',550)+text(cx,239,shapeText,12,COL.muted,'middle');return out;}
 $('shapeDrawing').innerHTML=`<svg viewBox="0 0 720 255" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${matrix(130,K,M,`K = ${fmt(p.K)}`,'M',p.left,`${fmt(p.M)} × ${fmt(p.K)}`,COL.k,COL.m)}${text(244,116,'×',25,COL.muted,'middle')}${matrix(362,N,K,`N = ${fmt(p.N)}`,'K',p.right,`${fmt(p.K)} × ${fmt(p.N)}`,COL.n,COL.k)}${text(486,116,'=',25,COL.muted,'middle')}${matrix(605,N,M,`N = ${fmt(p.N)}`,'M',p.output,`${fmt(p.M)} × ${fmt(p.N)}`,COL.n,COL.m)}</svg>`;
 $('shapeDrawing').setAttribute('aria-label',`${p.left} ${p.M}行${p.K}列，乘${p.right} ${p.K}行${p.N}列，得到${p.output} ${p.M}行${p.N}列`);
}
function renderCost(p){
 const c=cost(p),[fv,fu]=compact(c.flops),[bv,bu]=compact(c.total,1024);
 $('aiValue').textContent=c.ai.toLocaleString('en-US',{maximumFractionDigits:2});$('flopsValue').textContent=fv;$('flopsUnit').textContent=fu;$('bytesValue').textContent=bv;$('bytesUnit').textContent=bu;
 const colors=[COL.m,COL.k,COL.n],labels=['A 输入读取',`${p.family==='linear'||p.family==='moe'?'B 权重':'B 右矩阵'}读取`,'C 输出写入'];
 $('traffic').innerHTML=c.bytes.map((n,i)=>`<span style="width:${n/c.total*100}%;background:${colors[i]}" title="${labels[i]} ${(n/c.total*100).toFixed(1)}%"></span>`).join('');
 $('trafficLegend').innerHTML=c.bytes.map((n,i)=>`<span><i style="background:${colors[i]}"></i>${labels[i]} ${compact(n,1024).join(' ')} · ${(n/c.total*100).toFixed(1)}%</span>`).join('');
 $('dtypeNote').textContent=`每元素字节 sA / sB / sC = ${c.sizes.join(' / ')}。不乘字节数的原式得到 ${c.elementAI.toFixed(2)} FLOP / 元素；当前按一个 ${p.family==='moe'?'专家 ':''}GEMM 计算，未把额外批维相加。`;
 renderChart(p,c);
 const isWeights=['linear','moe'].includes(p.family);
 $('chartNote').textContent=isWeights?'固定当前 N、K，增加 M：一份权重被更多 token 复用。权重流量占主导时 AI 近似随 M 增长；输入输出逐渐占比更大，曲线会趋向有限上限。':'曲线仅演示“固定当前 N、K，再单独改变 M”的数学关系。当前 M=q；增加请求数 b 不会沿这条曲线移动，因为请求之间没有共享同一份 KV。移动上下文会改变 N 或 K，从而换一条曲线。';
 $('intensityTitle').textContent=isWeights?'同一份权重，能服务多少行？':'固定 N、K，只改变 M 会怎样？';
}
function renderChart(p,c){
 const width=720,height=265,left=62,right=24,top=35,bottom=47,maxM=8192;
 const x=m=>left+Math.log2(m)/13*(width-left-right),maxY=cost(p,maxM).ai*1.12,y=v=>height-bottom-v/maxY*(height-top-bottom);
 let contents=text(left,17,'AI · FLOP / byte',11,COL.muted)+text(width-right,17,`固定 N=${fmt(p.N)}，K=${fmt(p.K)}`,11,COL.muted,'end');
 for(let i=0;i<=4;i++){const val=maxY*i/4,yy=y(val);contents+=`<path d="M ${left} ${yy} H ${width-right}" stroke="${COL.line}"/>`+text(left-10,yy+4,val<10?val.toFixed(1):fmt(Math.round(val)),11,COL.muted,'end');}
 for(const m of [1,8,64,512,4096,8192])contents+=text(x(m),height-bottom+22,fmt(m),11,COL.muted,'middle');
 const points=Array.from({length:131},(_,i)=>{const m=2**(i/10);return [x(m),y(cost(p,m).ai)];});
 const line=points.map((a,i)=>`${i?'L':'M'}${a[0].toFixed(2)},${a[1].toFixed(2)}`).join(' ');
 contents+=`<path d="${line} L ${x(maxM)} ${height-bottom} L ${left} ${height-bottom} Z" fill="#edf3fc"/><path d="${line}" stroke="${COL.m}" stroke-width="2.8" fill="none"/>`;
 const cx=x(p.M),cy=y(c.ai),labelX=Math.max(left+48,Math.min(width-right-50,cx));
 contents+=`<path d="M ${cx} ${cy} V ${height-bottom}" stroke="${COL.m}" stroke-width="1" stroke-dasharray="4 4"/><circle cx="${cx}" cy="${cy}" r="5" fill="${COL.m}" stroke="white" stroke-width="2"/>`+text(labelX,Math.max(top+14,cy-14),`M=${fmt(p.M)} · AI=${c.ai.toFixed(1)}`,12,COL.m,'middle',650)+text((width+left-right)/2,height-5,'M · 对数坐标（每一步按倍数增长）',11,COL.muted,'middle');
 $('chart').innerHTML=`<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${contents}</svg>`;
 $('chart').setAttribute('aria-label',`固定N ${p.N}和K ${p.K}，M从1到8192的理想算术强度曲线。当前M ${p.M}，AI ${c.ai.toFixed(2)} FLOP每字节。`);
}
function render(){updateControls();const p=shape(state);$('shapeTitle').textContent=p.label;$('scope').textContent=p.scope;['m','n','k'].forEach((key,i)=>{$(`${key}Value`).textContent=fmt(p[key.toUpperCase()]);$(`${key}Meaning`).textContent=p.meaning[i];$(`${key}Change`).textContent=p.changes[i];});$('shapeNote').textContent=contextNote(p);$('attentionNote').hidden=!['attention','index'].includes(p.family);shapeDrawing(p);renderCost(p);}
function renderToy(){
 const t=toyStep(step);
 function mat(data,name,dim,type){let html=`<div class="toy-matrix"><div class="toy-heading">${name} <small>${dim}</small></div><div class="toy-bracket"><table aria-label="示例矩阵 ${name}"><tbody>`;for(let i=0;i<data.length;i++){html+='<tr>';for(let j=0;j<data[i].length;j++){const active=!t.done&&((type==='a'&&i===t.i&&j===t.k)||(type==='b'&&i===t.k&&j===t.j)||(type==='c'&&i===t.i&&j===t.j));const row=!t.done&&((type==='a'&&i===t.i)||(type==='b'&&j===t.j));const complete=type==='c'&&(i*3+j)<Math.floor(step/4);html+=`<td class="${active?'active':complete?'complete':row?'selected':''}">${data[i][j]}</td>`;}html+='</tr>';}return html+'</tbody></table></div></div>';}
 $('toyDrawing').innerHTML=`<div class="toy-matrices">${mat(TOY_A,'A','3 × 4','a')}<span class="toy-sign">×</span>${mat(TOY_B,'B','4 × 3','b')}<span class="toy-sign">=</span>${mat(t.out,'C','3 × 3','c')}</div>`;
 $('stepCount').textContent=`${step} / 36 次乘加`;
 if(t.done)$('calculation').innerHTML='<b>全部完成。</b> 3 × 3 个输出，每个沿 K=4 累加：36 次乘加 = 72 FLOPs。';
 else{const a=TOY_A[t.i][t.k],b=TOY_B[t.k][t.j],prev=t.out[t.i][t.j];$('calculation').innerHTML=`下一步：C[${t.i+1}, ${t.j+1}] = <b>${prev} + ${a} × ${b} = ${prev+a*b}</b><span class="toy-k">沿 K 的第 ${t.k+1} / 4 项</span>`;}
 $('toyDrawing').setAttribute('aria-label',t.done?'矩阵乘完成，共36次乘加72 FLOPs':`下一步计算第${t.i+1}行第${t.j+1}列，沿K第${t.k+1}项。当前累加值${t.out[t.i][t.j]}`);
 $('play').textContent=playing?'暂停':step===36?'重新播放':'播放乘法';$('play').setAttribute('aria-pressed',String(playing));$('next').disabled=t.done;
}
function stop(){playing=false;clearInterval(timer);timer=null;renderToy();}
function advance(){if(step<36)step++;if(step===36)stop();else renderToy();}
function start(){if(step===36)step=0;playing=true;clearInterval(timer);timer=setInterval(advance,Number($('speed').value));renderToy();}
function stopSweep(){clearInterval(sweepTimer);sweepTimer=null;$('sweep').textContent='▶ 自动增加 batch';$('sweep').setAttribute('aria-pressed','false');}
for(const [key,node] of Object.entries(el))node.addEventListener(['b','ctx'].includes(key)?'input':'change',()=>{stopSweep();state=normalize({...state,[key]:node.value});render();saveURL();});
$('play').addEventListener('click',()=>playing?stop():start());$('next').addEventListener('click',()=>{stop();advance();});$('restart').addEventListener('click',()=>{stop();step=0;renderToy();});$('speed').addEventListener('change',()=>{if(playing)start();});
$('sweep').addEventListener('click',()=>{if(sweepTimer){stopSweep();return;}state={...state,b:1};render();saveURL();$('sweep').textContent='Ⅱ 暂停 batch 演示';$('sweep').setAttribute('aria-pressed','true');const bs=[2,4,8,16,32,64,128,256,512];let pos=0;sweepTimer=setInterval(()=>{state={...state,b:bs[pos++]};render();saveURL();if(pos===bs.length)stopSweep();},850);});
$('share').addEventListener('click',async()=>{saveURL();try{await navigator.clipboard.writeText(location.href);$('shareStatus').textContent='已复制，可以分享当前配置。';}catch{$('shareStatus').textContent='配置已写入地址栏，可直接复制地址。';}});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();stopSweep();}});
window.addEventListener('popstate',()=>{stopSweep();state=normalize({...DEFAULT,...Object.fromEntries(new URLSearchParams(location.search))});render();});
render();renderToy();
