'use strict';
const $=id=>document.getElementById(id);let mutation=false,lastState=null;
function errorLabel(code){return ({task_busy:'任务仍在处理，请先停止并等待确认。',invalid_request:'请输入有效任务，不接受额外参数。',invalid_text:'请输入 1 到 2000 字的任务。',qwen_unavailable:'Qwen 服务不可用，请检查服务状态。',brain_unavailable:'Qwen 服务不可用。',jev_unavailable:'Jev 服务不可用，不能执行动作。',execution_unavailable:'执行服务尚未就绪。',provider_unavailable:'模型服务不可用。',camera_unavailable:'相机画面暂不可用。',request_rejected:'请求未被接受，请查看服务状态。'})[code]||String(code||'')}
function providerReady(provider){return Boolean(provider?.available??provider?.configured)}
function textValue(value){return typeof value==='string'?value:JSON.stringify(value)}
function eventText(event){
const d=event.detail||{};if(typeof d==='string')return errorLabel(d);
const names={observing:'获取完整相机图像',visual_decision:'Qwen 图像判断',decision_received:'动作决策已收到，尚不代表动作已施加',motion_applied:'动作指令已施加，运动与图像结果待确认',motion_completed:'本次微动作结束，不代表任务完成',visual_judgment:'Qwen 图像判断结果（不是独立几何验证）',stopped:'运行已停止',stopping:'正在停止',cycle_review:'暂停微动作，Qwen 根据新图像复核'};
const parts=[names[d.phase]||d.phase||event.state||'进度更新'];
for(const [key,label] of Object.entries({observation:'观察',instruction:'任务指引',goal:'当前目标',action:'动作',decision:'结构化判断',accepted:'决策是否通过接收检查',applied:'动作是否已施加',confidence:'置信度',reason:'原因',provenance:'来源',calls:'模型请求数',actions:'已应用动作数',iteration:'轮次',frame_id:'图像帧'}))if(d[key]!==undefined)parts.push(label+'：'+textValue(d[key]));
return parts.join(' · ');
}
function outcomeText(workflow){
const outcome=workflow.outcome||{};
if(workflow.state==='completed'&&outcome.model_assessed_complete===true&&outcome.completion_basis==='qwen_visual_assessment')return 'Qwen 根据图像判断任务已完成（模型判断；未经独立几何验证）。';
if(workflow.state==='completed')return '本次流程已结束；未收到模型确认任务完成的证据。';
if(['failed','blocked','cancelled','stopped'].includes(workflow.state))return '任务已停止或受阻，未确认完成。'+(outcome.reason?' 原因：'+errorLabel(outcome.reason):'');
return '尚未确认任务完成。';
}
function sessionBudget(label,provider){
const parts=[];
if(Number.isFinite(provider?.remaining_calls))parts.push(`剩余 ${provider.remaining_calls} 次`);
if(Number.isFinite(provider?.remaining_seconds))parts.push(`剩余 ${Math.max(0,Math.ceil(provider.remaining_seconds))} 秒（空闲也计时）`);
return parts.length?label+' 会话：'+parts.join('，'):label+' 会话预算未提供';
}
function render(state){
lastState=state;const w=state.workflow||{},q=state.qwen||{},j=state.jev||{};
const direct=w.control_mode==='qwen_direct';
$('control-mode').textContent=direct?'Qwen 直接视觉控制 · Qwen 选择动作，Jev 未启用':w.control_mode==='two_model'?'双模型控制 · Qwen 观察，Jev 选择动作':'控制模式尚未提供';
const active=Boolean(w.pending||['queued','observing','planning','executing','replanning','reviewing','stopping'].includes(w.state));
$('request').disabled=active||mutation;$('task-send').disabled=active||mutation||!w.execution_available||!providerReady(q);$('chat-send').disabled=active||mutation||!providerReady(q);$('stop').disabled=!active||mutation;$('reset').disabled=active||mutation;
$('input-status').textContent=active?'任务处理中，输入已锁定；请先停止并等待确认。':'Enter 提交执行任务，Shift + Enter 换行。';
$('readiness').textContent=w.execution_available&&providerReady(q)?'任务执行服务已就绪。提交后将根据完整图像尝试执行。':'当前不能执行任务：'+errorLabel(w.unavailable_reason||(!providerReady(q)?'qwen_unavailable':'execution_unavailable'))+' 没有任务排队。';
const labels={idle:'尚未提交任务',queued:'任务已提交',observing:'正在观察相机图像',planning:'Qwen 正在理解任务',executing:'正在执行模型选择的有界动作',replanning:'Qwen 正在复核新图像',reviewing:'Qwen 正在复核新图像',blocked:'任务受阻',failed:'任务失败',cancelled:w.pending?'正在停止，等待待处理调用退出':'任务已停止',preview:'仅交流结果，无动作执行',clarification:'需要澄清，未继续动作',stopped:'Qwen 要求停止',completed:'流程已结束',stopping:'正在停止'};
$('request-id').textContent=w.id?'请求 '+w.id.slice(0,8):'';$('workflow-status').textContent=(labels[w.state]||w.state||'')+(w.error?' · '+errorLabel(w.error):'');
const counts=[];for(const [key,label] of Object.entries({qwen_calls:'Qwen 请求',decision_calls:'动作决策请求',controller_calls:'实际 Jev 请求',controller_actions:'已施加动作',cycle_index:'周期索引'}))if(Number.isFinite(w[key]))counts.push(label+' '+w[key]);$('budgets').textContent='本任务请求记录：'+(counts.join(' · ')||'暂无')+'。 '+sessionBudget('Qwen',q)+'；'+(direct?'Jev 未启用':sessionBudget('Jev',j));
$('chat-history').replaceChildren();for(const m of w.history||[]){const p=document.createElement('p');p.className='message';p.textContent=(m.role==='user'?'你：':'Qwen 输出（不是系统执行状态）：')+(m.content||m.text||'');$('chat-history').appendChild(p)}
$('runtime-notices').replaceChildren();for(const n of w.runtime_notices||[]){const p=document.createElement('p');p.className='notice';p.textContent='系统反馈 · '+(n.request_id||'')+' · '+(n.request_text||'')+' · '+errorLabel(n.reason||n.message||'请求未完成');$('runtime-notices').appendChild(p)}
$('events').replaceChildren();for(const event of w.events||[]){const li=document.createElement('li');li.textContent=eventText(event);$('events').appendChild(li)}
$('observation').textContent=w.observation?'最近观察：'+textValue(w.observation):'尚无模型图像观察';$('outcome').textContent=outcomeText(w);
$('robot').textContent=JSON.stringify(state.robot||{},null,2);const c=state.camera||{};$('camera-status').textContent='本地相机：'+(c.status||'等待')+(Number.isFinite(c.age_seconds)?' · 预览年龄 '+c.age_seconds.toFixed(1)+' 秒':'')+' · 本地采样不调用模型';
}
function renderFrames(packet){
for(const im of packet.images||[]){if(!['left','right'].includes(im.camera_id)||!im.data_url?.startsWith('data:image/'))continue;const m=(packet.view_metadata||[]).find(v=>v.camera_id===im.camera_id);const label=({observer:'固定观察相机',wrist:'腕部相机'})[m?.role]||'相机 '+im.camera_id;const c=m?.calibration;const fov=c&&Number.isFinite(c.height)&&Number.isFinite(c.fy)&&c.fy>0?2*Math.atan(c.height/(2*c.fy))*180/Math.PI:null;$('camera-'+im.camera_id).src=im.data_url;$('camera-'+im.camera_id+'-caption').textContent=label+' · 完整模拟 RGB'+(fov!==null?' · 垂直视场 '+fov.toFixed(1)+'°':'')}
}
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const v=await r.json();if(!r.ok)throw Error(errorLabel(typeof v.detail==='string'?v.detail:'request_rejected'));return v}
async function poll(){try{render(await api('/api/state'))}catch(e){$('error').textContent=e.message}}
async function frames(){if(document.hidden)return;try{renderFrames(await api('/api/frames'))}catch(e){$('camera-status').textContent=e.message}}
async function command(path,body){if(mutation)return;mutation=true;if(lastState)render(lastState);$('error').textContent='';try{render(await api(path,body))}catch(e){$('error').textContent=e.message}finally{mutation=false;await poll()}}
$('task-send').onclick=()=>command('/api/tasks',{text:$('request').value,execute:true});$('chat-send').onclick=()=>command('/api/tasks',{text:$('request').value,execute:false});$('stop').onclick=()=>command('/api/control',{action:'stop'});$('reset').onclick=()=>command('/api/control',{action:'reset'});
$('request').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!$('task-send').disabled)$('task-send').click();else $('error').textContent='任务未提交：'+$('readiness').textContent}};
poll();frames();setInterval(poll,1000);setInterval(frames,1000);
