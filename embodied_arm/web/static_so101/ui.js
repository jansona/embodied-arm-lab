'use strict';
const $=id=>document.getElementById(id);let mutation=false;
async function api(path,body){const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const result=await response.json();if(!response.ok)throw Error(typeof result.detail==='string'?formatTaskError(result.detail):'请求未完成');return result}
function formatAxisUncertainty(band){
const status={outside:'区间在容差外',inside:'估计区间位于容差内',indeterminate:'区间与边界重叠，精度未确定'};
if(!['x','y','z'].includes(band.axis)||!Array.isArray(band.interval_m)||band.interval_m.length!==2||!band.interval_m.every(Number.isFinite))return '诊断区间不可用';
return `${band.axis.toUpperCase()}：误差诊断区间 [${(band.interval_m[0]*1000).toFixed(2)}, ${(band.interval_m[1]*1000).toFixed(2)}] mm · ${status[band.status]||'状态未知'}`;
}
function formatTaskError(code){
const labels={jev_unavailable:'Jev 控制器会话不可用；执行不会启动，仍可仅对话。',controller_readiness_unknown:'无法确认 Jev 控制器是否就绪；执行不会启动，仍可仅对话。',visual_execution_not_validated:'自动执行尚未验证，可使用仅对话。',brain_unavailable:'Qwen 对话服务离线，本次请求未发送。',execution_unavailable:'自动执行尚未验证；没有执行动作，可选择仅对话。',task_busy:'已有任务正在处理，本次请求未发送。',invalid_text:'请输入 1 到 2000 字的消息。',observation_unavailable:'无法取得有效视觉观察，没有执行动作。',planning_failed:'Qwen 请求失败或规划未通过校验，没有执行动作。',brain_rejected:'Qwen 回复未通过校验，没有执行动作。',unexpected_motion_plan:'仅对话模式收到动作计划，已拒绝；没有执行动作。',unsupported_task:'当前执行器不支持该任务，没有执行动作。',execution_failed:'执行途中失败，机械臂已停止；任务未完成。',execution_unverified:'动作结果未通过验证，机械臂已停止；不能确认任务完成。'};
return labels[code]||code||'';
}
function formatEventDetail(detail){
const labels={conversation_only:'仅对话，未执行动作',visually_verified:'结果验证通过（请结合来源标记判断）'};
return labels[detail]||formatTaskError(detail);
}
function render(s){const r=s.robot,p=s.perception;
$('robot-name').textContent=r.name||'SO-ARM101';$('robot-status').textContent=r.paused?'已停止':r.busy?'动作执行中':'空闲';
$('ee').textContent=Array.isArray(r.ee)?r.ee.map(x=>Number(x).toFixed(4)).join(' / ')+' m':'不可用';
$('joints').textContent=Array.isArray(r.joints)?r.joints.map(x=>Number(x).toFixed(3)).join(' / '):'不可用';
$('gripper').textContent=r.gripper_openness==null?'不可用':Number(r.gripper_openness).toFixed(3);
const labels={ready:'图像估计已通过检查',observing:'正在识别冻结的双目图像…',vision_unconfigured:'视觉模块未配置',stale_or_invalidated:'观察已过期或被动作失效，请重新识别',vision_unavailable_or_uncertain:'视觉不可用或不确定，拒绝生成可执行坐标',cancelled:'观察已取消',occluded:'目标遮挡，拒绝规划',uncertain_detection:'检测不确定，拒绝规划',landmark_uncertain:'三角测量未通过不确定性检查',frame_mismatch:'检测与相机帧不匹配',scene_changed_during_vision:'识别期间场景改变，结果已丢弃'};
$('vision-status').textContent=labels[p.status]||p.status;
const fixture=String(p.provenance||'').includes('fixture')||String(p.last_vision?.base_detector_provenance||'').includes('fixture');
$('perception-method').textContent=p.rgb_refinement_enabled?(fixture?'离线测试粗定位 + RGB顶面几何细化':'Qwen粗定位 + RGB顶面几何细化'):'未启用 RGB 几何细化';
$('vision-source').textContent=`来源：${p.provenance||'none'} · 初始检测：${p.last_vision?.base_detector_provenance||'无结果'} · ${p.image_source} · 不使用真值回退`;
$('refinement-details').hidden=!p.last_vision?.refinement;$('refinement-evidence').textContent=p.last_vision?.refinement?JSON.stringify(p.last_vision.refinement,null,2):'';
$('uncertainty').textContent=p.fresh?`图像年龄 ${Number(p.age_seconds).toFixed(1)} 秒 · 最大允许位置标准差 ${Number(p.max_position_std_m*1000).toFixed(1)} mm · 仅冻结场景 · 检测图像 ${Object.values(p.scene?.frame_ids||{}).join(' / ')}`:(p.preview_only?'以下是已过期的历史估计，仅供查看，不能用于规划或动作。':'尚无可用于规划的检测。遮挡、低置信度、过期或高不确定性会停止。')+(p.last_vision?.landmark_rejections?' '+JSON.stringify(p.last_vision.landmark_rejections):'');
$('objects').replaceChildren();for(const [name,o] of Object.entries(p.scene?.objects||{})){const li=document.createElement('li');li.textContent=`${name} · ${o.status} · ${o.landmark} · ${o.xyz?o.xyz.map(x=>Number(x).toFixed(4)).join(' / ')+' m':'无估计'} · σ ${o.position_std_m==null?'未知':(o.position_std_m*1000).toFixed(2)+' mm'} · ${o.reason||''}`;$('objects').appendChild(li)}
$('task-status').textContent=`独立动作计划预览（此预览不执行） · ${s.task.plan&&!s.task.scene_fresh?'历史预览（观察已失效）':s.task.status} · 来源 ${s.task.provenance||'无计划'}`;
$('model-status').textContent=`Qwen 大脑：${s.brain.configured?'已配置':'未配置'}，${s.brain.calls||0} 次 · RGB 视觉：${p.worker?.calls||0} 次，剩余 ${p.worker?.remaining_calls??0} 次 · Jev：${s.waypoint?.motion_enabled?(s.controller.available?'实验会话可用':'会话不可用'):'运动未启用'} · 无自动回退`;
const chat=s.chat||{};$('send').disabled=!chat.available||s.pending||mutation;$('plan-send').disabled=!s.task.available||s.pending||mutation;
const chatLabels={queued:'请求已排队',observing:'正在获取视觉观察…',executing:'任务正在执行，请查看进度',blocked:'任务已被阻止',idle:'可以发送文字消息',thinking:'Qwen 正在回复…',ready:'对话回复已收到',cancelled:'对话已取消，迟到回复不会应用',error:'对话未完成'};
$('chat-status').textContent=formatTaskError(chat.error)||(!chat.available?'Qwen 对话服务离线（尚未启动或会话已结束），不是视觉缺失。你仍可输入消息；服务恢复后再发送。':chatLabels[chat.status]||chat.status);
if(chat.provenance&&chat.provenance!=='none')$('chat-status').textContent+=' · 回复来源：'+(chat.provenance==='live'?'真实 Qwen API':(/mock|fake|fixture/.test(chat.provenance)?'离线测试（'+chat.provenance+'）':chat.provenance));
const flow=s.workflow||{};const flowLabels={idle:'尚未提交任务',queued:'任务已排队',observing:'正在获取新鲜视觉观察',planning:'Qwen 正在理解与规划',executing:'机械臂正在执行',completed:flow.execute?'任务流程结束':'对话结束',blocked:'任务被阻止',failed:'任务失败',cancelled:'任务已停止'};
$('task-send').disabled=!flow.execution_available||!chat.available||s.pending||mutation;
$('workflow-status').textContent=(flowLabels[flow.state]||flow.state||'')+(flow.error?' · '+formatTaskError(flow.error):'')+(!flow.execution_available?' · '+formatTaskError(flow.unavailable_reason||'visual_execution_not_validated'):'');
$('workflow-events').replaceChildren();for(const event of flow.events||[]){const li=document.createElement('li');li.textContent=(flowLabels[event.state]||event.state)+(event.detail?' · '+(typeof event.detail==='string'?formatEventDetail(event.detail):JSON.stringify(event.detail)):'');$('workflow-events').appendChild(li)}
$('chat-history').replaceChildren();for(const message of chat.messages||[]){const row=document.createElement('p');row.className='chat-message '+message.role;row.textContent=(message.role==='user'?'你：':'大脑回复：')+message.content;$('chat-history').appendChild(row)}
$('perceive').disabled=!p.configured||s.pending||mutation;
const w=s.waypoint||{};for(const id of ['waypoint-red','waypoint-blue'])$(id).disabled=!w.available||s.pending||mutation;$('waypoint-status').textContent=w.motion_enabled?`实验航点 · ${w.status} · 最多 ${w.max_calls} 次 Jev / ${w.max_seconds} 秒；1 次 Qwen 语义定位后逐步本地 RGB 跟踪（60秒/30帧/4px漂移限制），不代表抓取成功`:'执行接口等待验证，暂不可用。不会发送 Jev 请求。';
const waypointReasons={uncertainty_unresolved:'当前视觉精度无法判断是否进入 ±3 mm 容差，已停止；尚未确认到达。',estimated_uncertainty_envelope_within_tolerance:'估计误差区间符合 ±3 mm 判定条件；这不是经过校准的到达保证。',controller_rejected:'Jev 决策未通过接受条件，已停止。'};
$('waypoint-reason').textContent=waypointReasons[w.reason]||w.reason||'';
$('waypoint-uncertainty').replaceChildren();const bands=Array.isArray(w.axis_uncertainty)?w.axis_uncertainty:[];
for(const band of bands){const li=document.createElement('li');li.textContent=formatAxisUncertainty(band);$('waypoint-uncertainty').appendChild(li)}
$('waypoint-uncertainty-note').hidden=!bands.length;
$('plan-preview').hidden=!s.task.plan;$('plan-preview').textContent=s.task.plan?JSON.stringify(s.task.plan,null,2):'';
$('error').textContent=s.error||''}

async function state(){try{render(await api('/api/state'))}catch(e){$('error').textContent=e.message}}
async function frames(){if(document.hidden)return;try{const f=await api('/api/frames');for(const image of f.images){if(['left','right'].includes(image.camera_id)&&image.data_url.startsWith('data:image/'))$('camera-'+image.camera_id).src=image.data_url}$('frame-label').textContent='同步图像组 '+f.frame_id+' · 仅预览，未识别'}catch(e){$('error').textContent=e.message}}
async function control(action){if(mutation)return;mutation=true;$('reset').disabled=true;try{render(await api('/api/control',{action,seed:0}));if(action==='reset')await frames()}catch(e){$('error').textContent=e.message}finally{mutation=false;$('reset').disabled=false}}
async function request(path,body){if(mutation)return;mutation=true;try{render(await api(path,body||{}))}catch(e){$('error').textContent=e.message}finally{mutation=false;await state()}}
$('waypoint-red').onclick=()=>request('/api/waypoint',{object_id:'red'});$('waypoint-blue').onclick=()=>request('/api/waypoint',{object_id:'blue'});
$('task-send').onclick=()=>request('/api/tasks',{text:$('request').value,execute:true});
$('perceive').onclick=()=>request('/api/observe',{});$('send').onclick=()=>request('/api/chat',{text:$('request').value});$('plan-send').onclick=()=>request('/api/plan',{text:$('request').value});
$('request').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!$('send').disabled)$('send').click()}};
$('stop').onclick=()=>control('stop');$('reset').onclick=()=>control('reset');state();frames();setInterval(state,1000);setInterval(frames,1500);
