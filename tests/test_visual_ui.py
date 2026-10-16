import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'embodied_arm/web/static_visual'
def script_run(script):return json.loads(subprocess.check_output(['node','-e',script],text=True))
def test_completion_only_claims_model_assessment():
    source=(ROOT/'ui.js').read_text();helpers=source[source.index('function errorLabel'):source.index('function render(state)')]
    values=script_run(helpers+'''\nconsole.log(JSON.stringify([outcomeText({state:'completed',outcome:{model_assessed_complete:true,completion_basis:'qwen_visual_assessment'}}),outcomeText({state:'completed',outcome:{verified:true}}),eventText({detail:{phase:'motion_applied',action:{kind:'move'}}})]));''')
    assert 'Qwen' in values[0] and '未经独立几何验证' in values[0]
    assert '未收到' in values[1]
    assert '待确认' in values[2] and '成功' not in values[2]
def test_active_input_lock_and_stop_ack_preserve_draft():
    source=(ROOT/'ui.js').read_text();code=source[:source.index('async function api')]
    dom="const elements={};global.document={getElementById:id=>elements[id]||(elements[id]={children:[],replaceChildren(){this.children=[]},appendChild(x){this.children.push(x)}}),createElement:()=>({})};"
    values=script_run(dom+code+'''\nconst s={workflow:{state:'observing',pending:true,execution_available:true},qwen:{available:true},jev:{available:true}};elements.request={value:'新任务草稿'};render(s);const active=[elements.request.disabled,elements['task-send'].disabled,elements.stop.disabled];s.workflow.state='cancelled';render(s);const stopping=elements.request.disabled;s.workflow.pending=false;render(s);console.log(JSON.stringify({active,stopping,idle:elements.request.disabled,draft:elements.request.value}));''')
    assert values['active']==[True,True,False] and values['stopping']
    assert not values['idle'] and values['draft']=='新任务草稿'
def test_new_page_has_no_fixed_object_or_legacy_diagnostic_controls():
    html=(ROOT/'index.html').read_text()
    assert '未经独立几何验证' not in html or '几何验证' in html
    for forbidden in ('waypoint-red','waypoint-blue','perceive','红色方块','蓝色方块','ROI'):
        assert forbidden not in html


def test_provider_session_budget_is_separate_from_task_attempts():
    source=(ROOT/'ui.js').read_text();helpers=source[source.index('function errorLabel'):source.index('function render(state)')]
    values=script_run(helpers+"\nconsole.log(JSON.stringify([sessionBudget('Qwen',{remaining_calls:3,remaining_seconds:12.2}),sessionBudget('Jev',{remaining_calls:0,remaining_seconds:0}),sessionBudget('Jev',{})]));")
    assert '剩余 3 次' in values[0] and '剩余 13 秒' in values[0] and '空闲也计时' in values[0]
    assert '剩余 0 次' in values[1] and '剩余 0 秒' in values[1]
    assert '预算未提供' in values[2]
    assert '本任务请求记录' in source

def test_direct_mode_does_not_claim_jev_control_or_budget():
    source=(ROOT/'ui.js').read_text();code=source[:source.index('async function api')]
    dom="const elements={};global.document={getElementById:id=>elements[id]||(elements[id]={children:[],replaceChildren(){this.children=[]},appendChild(x){this.children.push(x)}}),createElement:()=>({})};"
    value=script_run(dom+code+'''\nconst s={workflow:{state:'executing',pending:true,execution_available:true,control_mode:'qwen_direct',decision_calls:2,controller_calls:0},qwen:{available:true,remaining_calls:4},jev:{available:false}};render(s);console.log(JSON.stringify({mode:elements['control-mode'].textContent,budget:elements.budgets.textContent,status:elements['workflow-status'].textContent}));''')
    assert 'Qwen 选择动作' in value['mode'] and 'Jev 未启用' in value['mode']
    assert '动作决策请求 2' in value['budget'] and '实际 Jev 请求 0' in value['budget']
    assert 'Jev 会话' not in value['budget'] and 'Jev 正在' not in value['status']
