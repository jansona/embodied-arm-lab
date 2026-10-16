"""Offline JavaScript formatter checks; no browser or model requests."""
import json
import subprocess
from pathlib import Path


def test_uncertainty_intervals_are_diagnostic_and_unresolved_is_not_success():
    source=(Path(__file__).resolve().parents[1]/'embodied_arm/web/static_so101'/'ui.js').read_text()
    helper=source[source.index('function formatAxisUncertainty'):source.index('function render(s)')]
    script=helper+"\nconsole.log(JSON.stringify([formatAxisUncertainty({axis:'x',interval_m:[-.004,.006],status:'indeterminate'}),formatAxisUncertainty({axis:'y',interval_m:[.004,.008],status:'outside'}),formatAxisUncertainty({axis:'z',interval_m:[-.002,.002],status:'inside'}),formatAxisUncertainty({axis:'x',interval_m:[NaN,0],status:'inside'})]));"
    values=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert '[-4.00, 6.00] mm' in values[0] and '精度未确定' in values[0]
    assert '容差外' in values[1] and '估计区间' in values[2]
    assert values[3]=='诊断区间不可用'
    assert '尚未确认到达' in source
    html=(Path(__file__).resolve().parents[1]/'embodied_arm/web/static_so101'/'index.html').read_text()
    assert '2σ + 1 mm' in html and '未校准为置信保证' in html and '±3 mm' in html


def test_error_labels_distinguish_no_motion_from_interrupted_execution():
    source=(Path(__file__).parent.parent/'embodied_arm/web/static_so101'/'ui.js').read_text()
    helper=source[source.index('function formatTaskError'):source.index('function render(s)')]
    script=helper+"\nconsole.log(JSON.stringify(['planning_failed','observation_unavailable','execution_failed','execution_unverified'].map(formatTaskError)));"
    values=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert all('没有执行动作' in v for v in values[:2])
    assert all('没有执行动作' not in v for v in values[2:])
    assert '已停止' in values[2] and '不能确认' in values[3]


def test_known_workflow_details_are_readable_and_provenance_aware():
    source=(Path(__file__).parent.parent/'embodied_arm/web/static_so101'/'ui.js').read_text()
    helper=source[source.index('function formatTaskError'):source.index('function render(s)')]
    script=helper+"\nconsole.log(JSON.stringify(['conversation_only','visually_verified','planning_failed'].map(formatEventDetail)));"
    values=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert '未执行动作' in values[0] and '来源' in values[1] and '没有执行动作' in values[2]
    assert "executing:'任务正在执行，请查看进度'" in source


def test_automatic_capture_is_distinct_from_paid_recognition_and_manual_is_optional():
    root=Path(__file__).parent.parent/'embodied_arm/web/static_so101'
    source=(root/'ui.js').read_text()
    html=(root/'index.html').read_text()
    helper=source[source.index('function formatTaskError'):source.index('function render(s)')]
    script=helper+"\nconsole.log(JSON.stringify([formatCameraStatus({status:'ready',interval_seconds:1,age_seconds:.2}),formatCameraStatus({status:'error',error:'camera_capture_failed'}),formatTaskError('vision_unavailable'),formatTaskError({code:'network_error'}),formatEventDetail('plan_preview')]));"
    values=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert '每 1 秒本地采样' in values[0] and '模型调用 0 次' in values[0]
    assert '采样失败' in values[1]
    assert all('无需上传图片' in v and '没有执行动作' in v for v in values[2:4])
    assert '未执行动作' in values[4]
    assert 'id="vision-diagnostics"' in html
    assert html.index('id="vision-diagnostics"') < html.index('id="perceive"')
    assert '发送（自动观察）' in html and '不调用付费模型' in html
    assert "setInterval(frames,1000)" in source
    assert "setInterval(()=>request('/api/observe'" not in source


def test_render_automatic_observation_error_and_plan_without_motion():
    source=(Path(__file__).parent.parent/'embodied_arm/web/static_so101'/'ui.js').read_text()
    code=source[:source.index('async function state()')]
    dom="const elements={};global.document={getElementById:(id)=>elements[id]||(elements[id]={replaceChildren(){},appendChild(){}}),createElement:()=>({})};"
    state={'robot':{},'perception':{'status':'stale_or_invalidated','worker':{}},'brain':{},'controller':{},'task':{'available':False},'chat':{'available':True,'status':'error','error':'network_error','messages':[]},'workflow':{'state':'blocked','error':'network_error','execution_available':False},'camera':{'status':'ready','interval_seconds':1},'pending':False}
    script=dom+code+'\nrender('+json.dumps(state)+');console.log(JSON.stringify({capture:elements["capture-status"].textContent,vision:elements["vision-status"].textContent,chat:elements["chat-status"].textContent,sendDisabled:elements.send.disabled,executeDisabled:elements["task-send"].disabled}));'
    value=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert '自动更新' in value['vision'] and '模型调用 0 次' in value['capture']
    assert '网络请求失败' in value['chat'] and '无需上传图片' in value['chat']
    assert value['sendDisabled'] is False and value['executeDisabled'] is True
    state['workflow'].update(state='plan_preview',error=None,plan={'action':'pick','object_id':'red'})
    state['chat'].update(status='plan_preview',error=None)
    script=dom+code+'\nrender('+json.dumps(state)+');console.log(JSON.stringify({status:elements["workflow-status"].textContent,preview:elements["plan-preview"].textContent,hidden:elements["plan-preview"].hidden}));'
    value=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert '未执行动作' in value['status'] and 'red' in value['preview'] and not value['hidden']


def test_main_task_plan_is_visible_and_stop_tracks_active_work():
    root=Path(__file__).parent.parent/'embodied_arm/web/static_so101'
    source=(root/'ui.js').read_text();html=(root/'index.html').read_text()
    assert html.index('id="structured-plan"') < html.index('id="advanced-diagnostics"')
    assert 'const preview=flow.plan;' in source and 'flow.plan||s.task.plan' not in source
    assert html.index('id="diagnostic-plan-preview"') > html.index('id="advanced-diagnostics"')
    assert html.index('id="advanced-diagnostics"') < html.index('id="waypoint-red"')
    code=source[:source.index('async function state()')]
    dom="const elements={};global.document={getElementById:(id)=>elements[id]||(elements[id]={children:[],replaceChildren(){this.children=[]},appendChild(v){this.children.push(v)}}),createElement:()=>({})};"
    state={'robot':{'busy':False},'perception':{'status':'ready','fresh':True,'worker':{}},'brain':{},'controller':{},'task':{},'chat':{'available':True,'status':'plan_preview','messages':[]},'workflow':{'state':'plan_preview','plan':{'action':'pick','object_id':'red'},'execution_available':False},'pending':False}
    script=dom+code+'\nconst s='+json.dumps(state)+';render(s);const idle={stop:elements.stop.disabled,plan:elements["structured-plan"].children[0].textContent,status:elements["plan-status"].textContent,execution:elements["execution-status"].textContent};s.pending=true;s.workflow.state="observing";render(s);console.log(JSON.stringify({idle,activeStop:elements.stop.disabled}));'
    value=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert value['idle']['stop'] and not value['activeStop']
    assert value['idle']['plan']=='夹起红色方块'
    assert '未执行动作' in value['idle']['status'] and '尚未执行' in value['idle']['execution']
    state['workflow'].update(state='failed',execute=True,motion_executed=True,events=[{'state':'executing','detail':{'action':'x_plus'}}])
    script=dom+code+'\nrender('+json.dumps(state)+');console.log(JSON.stringify({plan:elements["plan-status"].textContent,execution:elements["execution-status"].textContent}));'
    value=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert '未执行动作' not in value['plan'] and '中断' in value['plan']
    assert '未确认任务完成' in value['execution']
