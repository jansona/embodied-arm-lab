'use strict';
const $=id=>document.getElementById(id);let mutation=false;
async function api(path,body){const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const result=await response.json();if(!response.ok)throw Error(typeof result.detail==='string'?formatTaskError(result.detail):'请求未完成');return result}
function formatAxisUncertainty(band){
const status={outside:'区间在容差外',inside:'估计区间位于容差内',indeterminate:'区间与边界重叠，精度未确定'};
if(!['x','y','z'].includes(band.axis)||!Array.isArray(band.interval_m)||band.interval_m.length!==2||!band.interval_m.every(Number.isFinite))return '诊断区间不可用';
return `${band.axis.toUpperCase()}：误差诊断区间 [${(band.interval_m[0]*1000).toFixed(2)}, ${(band.interval_m[1]*1000).toFixed(2)}] mm · ${status[band.status]||'状态未知'}`;
}
function formatTaskError(code){
if(code&&typeof code==='object')code=code.code||code.reason||'observation_unavailable';
const labels={timeout:'视觉请求超时，未获得有效观察；没有执行动作。',expired:'视觉会话已到期，自动观察未完成；没有执行动作。',call_budget_exhausted:'视觉请求预算已用完，自动观察未完成；没有执行动作。',landmark_uncertain:'图像定位不够精确，不能安全生成动作；没有执行动作。',occluded:'目标被遮挡，未获得可靠观察；没有执行动作。',occluded_or_missing_feature:'目标被遮挡或缺少可识别特征；没有执行动作。',uncertain_detection:'图像检测不确定；没有执行动作。',low_confidence:'视觉识别置信度不足；没有执行动作。',unavailable:'视觉服务当前不可用；没有执行动作。',scene_changed_during_vision:'识别期间场景改变，结果已丢弃；没有执行动作。',invalid_or_stale_frame:'识别图像无效或已过期；没有执行动作。',stale_frame:'识别图像已过期；没有执行动作。',vision_unavailable:'视觉服务未启动或已到期，自动识别未能开始；没有执行动作，无需上传图片。',network_error:'视觉服务网络请求失败，自动观察未完成；没有执行动作，无需上传图片。',vision_network_error:'视觉服务网络请求失败，自动观察未完成；没有执行动作，无需上传图片。',camera_capture_failed:'模拟相机采样失败；没有执行动作，请查看相机状态。',jev_unavailable:'Jev 控制器会话不可用；执行不会启动，仍可发送消息或生成计划预览。',controller_readiness_unknown:'无法确认 Jev 控制器是否就绪；执行不会启动，仍可发送消息或生成计划预览。',visual_execution_not_validated:'自动执行尚未验证，可先发送消息或生成计划预览。',brain_unavailable:'Qwen 对话服务离线，本次请求未发送。',execution_unavailable:'自动执行尚未验证；没有执行动作，可先发送消息或生成计划预览。',task_busy:'已有任务正在处理，本次请求未发送。',invalid_text:'请输入 1 到 2000 字的消息。',observation_unavailable:'无法取得有效视觉观察，没有执行动作。',planning_failed:'Qwen 请求失败或规划未通过校验，没有执行动作。',brain_rejected:'Qwen 回复未通过校验，没有执行动作。',unexpected_motion_plan:'仅对话模式收到动作计划，已拒绝；没有执行动作。',unsupported_task:'当前执行器不支持该任务，没有执行动作。',execution_failed:'执行途中失败，机械臂已停止；任务未完成。',execution_unverified:'动作结果未通过验证，机械臂已停止；不能确认任务完成。'};
return labels[code]||code||'';
}
function formatEventDetail(detail){
const labels={execution_not_requested:'计划已生成，尚未请求执行',plan_preview:'动作计划已生成，仅供预览，未执行动作',observation_cached:'已复用有效视觉观察',observation_refresh:'正在自动识别当前双目图像',conversation_only:'仅对话，未执行动作',visually_verified:'结果验证通过（请结合来源标记判断）'};
return labels[detail]||formatTaskError(detail);
}
function formatPlanStep(step){
const objects={red:'红色方块',blue:'蓝色方块'},targets={goal:'目标区域','on:red':'红色方块上方','on:blue':'蓝色方块上方'};
const object=objects[step.object_id]||step.object_id||'目标';
if(step.action==='pick')return `夹起${object}`+(step.target_id?`，移至${targets[step.target_id]||step.target_id}`:'');
if(step.action==='place')return `放置${object}到${targets[step.target_id]||step.target_id||'目标区域'}`;
return {clarify:'需要进一步澄清',status:'状态说明',stop:'停止请求'}[step.action]||'等待有效计划';
}
function formatCameraStatus(camera){
if(!camera)return '自动相机状态暂不可用';
const labels={ready:'本地双目图像已更新',capturing:'正在本地采样双目图像',error:'本地相机采样失败',stopped:'本地相机采样已停止'};
return `${labels[camera.status]||'等待相机采样'} · 每 ${camera.interval_seconds||1} 秒本地采样 · 模型调用 0 次`+(Number.isFinite(camera.age_seconds)?` · 图像年龄 ${camera.age_seconds.toFixed(1)} 秒`:'')+(camera.error?' · '+formatTaskError(camera.error):'');
}
function render(s){const r=s.robot,p=s.perception;
$('robot-name').textContent=r.name||'SO-ARM101';$('robot-status').textContent=r.paused?'已停止':r.busy?'动作执行中':'空闲';
$('ee').textContent=Array.isArray(r.ee)?r.ee.map(x=>Number(x).toFixed(4)).join(' / ')+' m':'不可用';
$('joints').textContent=Array.isArray(r.joints)?r.joints.map(x=>Number(x).toFixed(3)).join(' / '):'不可用';
$('gripper').textContent=r.gripper_openness==null?'不可用':Number(r.gripper_openness).toFixed(3);
const labels={ready:'图像估计已通过检查',observing:'正在识别冻结的双目图像…',vision_unconfigured:'视觉模块未配置',stale_or_invalidated:'识别结果已过期或失效；下次场景请求会自动更新，历史估计不能用于动作',vision_unavailable_or_uncertain:'视觉不可用或不确定，拒绝生成可执行坐标',cancelled:'观察已取消',occluded:'目标遮挡，拒绝规划',uncertain_detection:'检测不确定，拒绝规划',landmark_uncertain:'三角测量未通过不确定性检查',frame_mismatch:'检测与相机帧不匹配',scene_changed_during_vision:'识别期间场景改变，结果已丢弃'};
$('capture-status').textContent=formatCameraStatus(s.camera);
$('vision-status').textContent=labels[p.status]||formatTaskError(p.status);
const fixture=String(p.provenance||'').includes('fixture')||String(p.last_vision?.base_detector_provenance||'').includes('fixture');
$('perception-method').textContent=p.rgb_refinement_enabled?(fixture?'离线测试粗定位 + RGB顶面几何细化':'Qwen粗定位 + RGB顶面几何细化'):'未启用 RGB 几何细化';
$('vision-source').textContent=`来源：${p.provenance||'none'} · 初始检测：${p.last_vision?.base_detector_provenance||'无结果'} · ${p.image_source} · 不使用真值回退`;
$('refinement-details').hidden=!p.last_vision?.refinement;$('refinement-evidence').textContent=p.last_vision?.refinement?JSON.stringify(p.last_vision.refinement,null,2):'';
$('uncertainty').textContent=p.fresh?`图像年龄 ${Number(p.age_seconds).toFixed(1)} 秒 · 最大允许位置标准差 ${Number(p.max_position_std_m*1000).toFixed(1)} mm · 仅冻结场景 · 检测图像 ${Object.values(p.scene?.frame_ids||{}).join(' / ')}`:(p.preview_only?'以下是已过期的历史估计，仅供查看，不能用于规划或动作。':'尚无可用于规划的检测。遮挡、低置信度、过期或高不确定性会停止。')+(p.last_vision?.landmark_rejections?' '+JSON.stringify(p.last_vision.landmark_rejections):'');
$('objects').replaceChildren();for(const [name,o] of Object.entries(p.scene?.objects||{})){const li=document.createElement('li');li.textContent=`${name} · ${o.status} · ${o.landmark} · ${o.xyz?o.xyz.map(x=>Number(x).toFixed(4)).join(' / ')+' m':'无估计'} · σ ${o.position_std_m==null?'未知':(o.position_std_m*1000).toFixed(2)+' mm'} · ${o.reason||''}`;$('objects').appendChild(li)}
$('task-status').textContent=`独立动作计划预览（此预览不执行） · ${s.task.plan&&!s.task.scene_fresh?'历史预览（观察已失效）':s.task.status} · 来源 ${s.task.provenance||'无计划'}`;
$('model-status').textContent=`Qwen 大脑：${s.brain.configured?'已配置':'未配置'}，${s.brain.calls||0} 次 · RGB 视觉：${p.worker?.calls||0} 次，剩余 ${p.worker?.remaining_calls??0} 次 · Jev：${s.waypoint?.motion_enabled?(s.controller.available?'实验会话可用':'会话不可用'):'运动未启用'} · 无自动回退`;
const chat=s.chat||{};$('send').disabled=!chat.available||s.pending||mutation;$('plan-send').disabled=!s.task.available||s.pending||mutation;
const chatLabels={queued:'请求已排队',observing:'正在获取视觉观察…',executing:'任务正在执行，请查看进度',blocked:'任务已被阻止',idle:'可以发送文字消息',thinking:'Qwen 正在回复…',ready:'回复已收到',plan_preview:'动作计划已生成，仅供预览，未执行动作',cancelled:'对话已取消，迟到回复不会应用',error:'对话未完成'};
$('chat-status').textContent=formatTaskError(chat.error)||(!chat.available?'Qwen 对话服务离线（尚未启动或会话已结束），不是视觉缺失。你仍可输入消息；服务恢复后再发送。':chatLabels[chat.status]||chat.status);
if(chat.provenance&&chat.provenance!=='none')$('chat-status').textContent+=' · 回复来源：'+(chat.provenance==='live'?'真实 Qwen API':(/mock|fake|fixture/.test(chat.provenance)?'离线测试（'+chat.provenance+'）':chat.provenance));
const flow=s.workflow||{};const flowLabels={idle:'尚未提交任务',queued:'任务已排队',observing:'正在获取新鲜视觉观察',planning:'Qwen 正在理解与规划',executing:'机械臂正在执行',plan_preview:'动作计划预览已生成，未执行动作',completed:flow.execute?'任务流程结束':'回复完成（未执行动作）',blocked:'任务被阻止',failed:'任务失败',cancelled:'任务已停止'};
$('task-send').disabled=!flow.execution_available||!chat.available||s.pending||mutation;
$('workflow-status').textContent=(flowLabels[flow.state]||flow.state||'')+(flow.error?' · '+formatTaskError(flow.error):'');
$('execution-readiness').textContent=flow.execution_available?'执行服务已就绪；只有点击发送并执行才会启动动作。':formatTaskError(flow.unavailable_reason||'visual_execution_not_validated');
$('workflow-events').replaceChildren();for(const event of flow.events||[]){const li=document.createElement('li');li.textContent=(flowLabels[event.state]||event.state)+(event.detail?' · '+(typeof event.detail==='string'?formatEventDetail(event.detail):JSON.stringify(event.detail)):'');$('workflow-events').appendChild(li)}
$('chat-history').replaceChildren();for(const message of chat.messages||[]){const row=document.createElement('p');row.className='chat-message '+message.role;row.textContent=(message.role==='user'?'你：':'大脑回复：')+message.content;$('chat-history').appendChild(row)}
$('perceive').disabled=!p.configured||s.pending||mutation;
const w=s.waypoint||{};for(const id of ['waypoint-red','waypoint-blue'])$(id).disabled=!w.available||s.pending||mutation;$('waypoint-status').textContent=w.motion_enabled?`实验航点 · ${w.status} · 最多 ${w.max_calls} 次 Jev / ${w.max_seconds} 秒；1 次 Qwen 语义定位后逐步本地 RGB 跟踪（60秒/30帧/4px漂移限制），不代表抓取成功`:'执行接口等待验证，暂不可用。不会发送 Jev 请求。';
const waypointReasons={uncertainty_unresolved:'当前视觉精度无法判断是否进入 ±3 mm 容差，已停止；尚未确认到达。',estimated_uncertainty_envelope_within_tolerance:'估计误差区间符合 ±3 mm 判定条件；这不是经过校准的到达保证。',controller_rejected:'Jev 决策未通过接受条件，已停止。'};
$('waypoint-reason').textContent=waypointReasons[w.reason]||w.reason||'';
$('waypoint-uncertainty').replaceChildren();const bands=Array.isArray(w.axis_uncertainty)?w.axis_uncertainty:[];
for(const band of bands){const li=document.createElement('li');li.textContent=formatAxisUncertainty(band);$('waypoint-uncertainty').appendChild(li)}
$('waypoint-uncertainty-note').hidden=!bands.length;
const active=Boolean(s.pending||r.busy||['running','executing','observing','planning'].includes(w.status));
$('stop').disabled=!active||mutation;
const preview=flow.plan;
const actionable=preview&&['pick','place','sequence'].includes(preview.action);
const planFresh=flow.plan_fresh??p.fresh;
$('diagnostic-plan-preview').hidden=!s.task.plan;$('diagnostic-plan-preview').textContent=s.task.plan?JSON.stringify(s.task.plan,null,2):'';
const executionStarted=Boolean(flow.motion_executed||(flow.events||[]).some(e=>e.state==='executing'));
const planProgress=flow.execute&&flow.state==='completed'?'执行流程已结束，请查看验证记录':flow.execute&&flow.state==='executing'?'正在执行该计划':executionStarted?'执行已中断或结束，请查看验证记录':'计划已生成，未执行动作';
$('plan-status').textContent=!preview?'尚无 Qwen 计划':actionable?planProgress+(planFresh?' · 视觉依据仍有效':' · 视觉依据已过期或失效，执行前必须重新观察'):'这是对话回复，不包含执行动作';
$('structured-plan').replaceChildren();
for(const step of preview?(preview.action==='sequence'?preview.steps||[]:[preview]):[]){const li=document.createElement('li');li.textContent=formatPlanStep(step);$('structured-plan').appendChild(li)}
const executionEvents=(flow.events||[]).filter(e=>e.state==='executing'||e.detail==='visually_verified'||['execution_failed','execution_unverified'].includes(e.detail));
$('execution-events').replaceChildren();for(const event of executionEvents){const li=document.createElement('li');li.textContent=typeof event.detail==='string'?formatEventDetail(event.detail):JSON.stringify(event.detail||{});$('execution-events').appendChild(li)}
$('execution-status').textContent=flow.state==='executing'?'Jev 执行中；尚未确认任务完成':flow.outcome?.verified===true?'执行结果已通过验证（请核对模型来源）':executionEvents.length?'执行已停止或结束；未确认任务完成':'尚未执行任何任务动作';
$('plan-preview').hidden=!preview;$('plan-preview').textContent=preview?JSON.stringify(preview,null,2):'';
$('error').textContent=formatTaskError(s.error)}

async function state(){try{render(await api('/api/state'))}catch(e){$('error').textContent=e.message}}
async function frames(){if(document.hidden)return;try{const f=await api('/api/frames');for(const image of f.images){if(['left','right'].includes(image.camera_id)&&image.data_url.startsWith('data:image/'))$('camera-'+image.camera_id).src=image.data_url}$('frame-label').textContent='相机图像组 '+f.frame_id+' · 本地 RGB 预览（识别状态见下方）'}catch(e){$('error').textContent=e.message}}
async function control(action){if(mutation)return;mutation=true;$('reset').disabled=true;try{render(await api('/api/control',{action,seed:0}));if(action==='reset')await frames()}catch(e){$('error').textContent=e.message}finally{mutation=false;$('reset').disabled=false}}
async function request(path,body){if(mutation)return;mutation=true;try{render(await api(path,body||{}))}catch(e){$('error').textContent=e.message}finally{mutation=false;await state()}}
$('waypoint-red').onclick=()=>request('/api/waypoint',{object_id:'red'});$('waypoint-blue').onclick=()=>request('/api/waypoint',{object_id:'blue'});
$('task-send').onclick=()=>request('/api/tasks',{text:$('request').value,execute:true});
$('perceive').onclick=()=>request('/api/observe',{});$('send').onclick=()=>request('/api/chat',{text:$('request').value});$('plan-send').onclick=()=>request('/api/plan',{text:$('request').value});
$('request').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!$('send').disabled)$('send').click()}};
$('stop').onclick=()=>control('stop');$('reset').onclick=()=>control('reset');state();frames();setInterval(state,1000);setInterval(frames,1000);
