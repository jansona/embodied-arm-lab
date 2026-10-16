import json
import httpx
import pytest
from adapters import JevAdapter
from microcontrol_experiment import MicroController,make_body

def response(choice='x_plus',confidence=.9,probabilities=None):
    return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':choice,'confidence':confidence,'probabilities':probabilities or {'x_plus':.9,'x_minus':.05,'hold':.05}}}})

def test_matched_candidates_and_geometry_only():
    for sign in (1,-1):
        bodies=[make_body('jev-latest',[.45,0,.3],[.45+.06*sign,0,.3],encoding) for encoding in ('numeric','relational')]
        assert bodies[0]['questions']==bodies[1]['questions']
        assert bodies[1]['state']['target_relative_to_gripper']==('right' if sign>0 else 'left')
        assert not any(a in json.dumps(bodies[1]['state']) for a in ('x_plus','x_minus','hold'))

def test_reuses_client_and_preserves_actual_choices():
    seen=[]
    def handler(request):seen.append(request);return response()
    c=MicroController(JevAdapter(api_key='test-placeholder'),httpx.MockTransport(handler));client=c.client
    try:
        for _ in range(2):
            decision,metadata=c.decide([.45,0,.3],[.51,0,.3])
            assert decision.accepted and decision.action=='x_plus'
            assert metadata['evaluated_candidates']==['x_plus','x_minus','hold']
            assert set(metadata['raw_probabilities'])==set(metadata['evaluated_candidates'])
            assert c.client is client
        assert len(seen)==2
    finally:c.close()
    assert client.is_closed

@pytest.mark.parametrize('payload',[response(confidence=.2),response(choice='x_minus'),response(probabilities={'x_plus':.7,'x_minus':.1,'hold':.1}),response(probabilities={'x_plus':True,'x_minus':0,'hold':0}),response(probabilities={'x_plus':1,'hold':0})])
def test_rejects_uncertain_or_invalid(payload):
    c=MicroController(JevAdapter(api_key='test-placeholder'),httpx.MockTransport(lambda r:payload))
    try:
        decision,_=c.decide([.45,0,.3],[.51,0,.3]);assert not decision.accepted
    finally:c.close()

def test_exclusive_inside_tolerance_criteria_and_multi_axis_rejection():
    from microcontrol_experiment import XMicroAdapter
    body=make_body('jev-latest',[.4501,0,.3],[.45,0,.3],distance_band=True)
    assert body['state']['measured_distance_band']=='inside_tolerance'
    assert 'outside_tolerance' in body['questions']['action']['criteria']['x_plus']
    assert 'inside_tolerance' in body['questions']['action']['criteria']['hold']
    seen=[]
    def handler(request):seen.append(request);return response()
    adapter=XMicroAdapter(JevAdapter(api_key='test-placeholder'),httpx.MockTransport(handler))
    try:
        result=adapter.decide({'objective':{'id':'local_target','target_xyz':[.5,.2,.3]},'end_effector':[.45,0,.3]})
        assert result.reason=='invalid_state' and not seen
    finally:adapter.close()

@pytest.mark.parametrize('axis',list('xyz'))
def test_all_axes_real_model_action_boundary(axis):
    from microcontrol_experiment import make_axis_body
    i='xyz'.index(axis);current=[.45,0,.3];target=current[:];target[i]+=.02
    body=make_axis_body('jev-latest',current,target,axis)
    keys={axis+'_plus',axis+'_minus','hold'}
    assert set(body['questions']['action']['criteria'])==keys
    def handler(request):
        return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':axis+'_plus','confidence':.99,'probabilities':{k:float(k==axis+'_plus') for k in keys}}}})
    c=MicroController(JevAdapter(api_key='test-placeholder'),httpx.MockTransport(handler))
    try:
        result,meta=c.decide(current,target,axis=axis)
        assert result.action==axis+'_plus' and result.accepted
        assert set(meta['evaluated_candidates'])==keys
    finally:c.close()

def test_phased_model_mock_physical_pick():
    import time
    from simulator import Simulation
    from task_executor import TaskExecutor,plan_from_intent
    from microcontrol_experiment import PhasedMicroAdapter
    seen=[]
    def handler(request):
        body=json.loads(request.content);state=body['state'];choices=body['questions']['action']['criteria'];seen.append(state)
        if 'axis_phase' in state:
            axis=state['axis_phase'];action='hold' if state['measured_distance_band']=='inside_tolerance' else axis+('_plus' if state['target_m'][axis]>state['current_m'][axis] else '_minus')
        else:
            desired=state['desired_gripper_state'];band=state['measured_gripper_band']
            action='hold' if (desired=='open' and band=='open') or (desired=='closed' and band=='closed_or_contact') else ('open' if desired=='open' else 'close')
        return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':action,'confidence':.99,'probabilities':{a:float(a==action) for a in choices}}}})
    adapter=PhasedMicroAdapter(JevAdapter(api_key='test-placeholder'),httpx.MockTransport(handler));sim=Simulation()
    try:
        executor=TaskExecutor(sim,adapter);executor.start(plan_from_intent({'action':'pick','object_id':'red'},sim.state()))
        deadline=time.monotonic()+20
        while executor.status()['status']=='running' and time.monotonic()<deadline:
            sim.tick(96);time.sleep(.001)
        assert executor.status()['status']=='completed',executor.status()
        assert sim.state()['blocks']['red'][2]>.13,sim.state()['blocks']
        assert executor.status()['controller_calls']<=48
        assert any(s.get('desired_gripper_state')=='closed' for s in seen)
    finally:adapter.close();sim.close()

@pytest.mark.parametrize('width,target,current_gap,target_gap',[(.08,'open','wide','wide'),(.08,'closed','wide','narrow'),(.05,'open','narrow','wide'),(.05,'closed','narrow','narrow')])
def test_gripper_transition_has_all_choices_and_observed_state(width,target,current_gap,target_gap):
    from microcontrol_experiment import PhasedMicroAdapter
    seen=[]
    def handler(request):
        body=json.loads(request.content);seen.append(body)
        return httpx.Response(200,json={'answers':{'action':{'type':'choice','choice':'hold','confidence':.99,'probabilities':{'open':0,'close':0,'hold':1}}}})
    adapter=PhasedMicroAdapter(JevAdapter(api_key='test-placeholder'),httpx.MockTransport(handler))
    try:
        decision=adapter.decide({'objective':{'id':'gripper','target_state':target},'gripper_width':width})
        body=seen[0];assert body['state']['current_finger_gap']==current_gap
        assert body['state']['target_finger_gap']==target_gap
        assert set(body['questions']['action']['criteria'])=={'open','close','hold'}
        assert decision.action=='hold' # never override the model with the table
    finally:adapter.close()
