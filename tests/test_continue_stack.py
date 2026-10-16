"""No-network tests for fixed HTTP actuator and exact-plan continuation guards."""
import copy
import hashlib
import json
import time
import httpx
import pytest
from embodied_arm.providers.adapters import Decision
from embodied_arm.legacy.panda.continue_stack import HTTPArmBackend, GuardedController, load_checkpoint


def scene():
    return {'engine':'PyBullet DIRECT / TinyRenderer','seed':0,'paused':True,'busy':False,
        'ee':[.48,.21,.05],'joints':[0.]*7,'blocks':{'red':[.48,.21,.038],'blue':[.58,-.19,.025]},
        'goal':[.48,.21,.025],'gripper_width':.05,'metrics':{'actions':59,'physics_steps':200},
        'workspace':[[.25,.7],[-.35,.35],[.022,.6]]}


def fixture_files(tmp_path):
    snapshot=scene()
    steps=[{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
           {'step_id':'s2','action':'place','object_id':'red','target_id':'goal'},
           {'step_id':'s3','action':'pick','object_id':'blue','target_id':None},
           {'step_id':'s4','action':'place','object_id':'blue','target_id':'on:red'}]
    plan={'action':'sequence','object_id':None,'target_id':None,'message':'红色作底蓝色在上','steps':steps}
    snapshot['chat']={'plan':plan,'controller':'jev','provenance':'live','task':{'steps':[
        {**s,'status':status,'outcome':{'verified':status=='completed'}} for s,status in zip(steps,['completed','failed','pending','pending'])]}}
    from embodied_arm.providers.adapters import ACTIONS
    probs={a:0. for a in ACTIONS};probs.update(open=.61,hold=.39)
    decision={'action':'hold','proposed_action':'open','accepted':False,'reason':'low_confidence',
              'confidence':.4,'probabilities':probs}
    snap=tmp_path/'snapshot.json';snap.write_text(json.dumps(snapshot))
    record={'plan':plan,'source_call_id':'test-live-plan','model':'qwen3.8-max','provenance':'live','accepted':True,
            'snapshot_sha256':hashlib.sha256(snap.read_bytes()).hexdigest(),
            'failed_objective':{'id':'gripper','target_state':'open'},'rejected_decision':decision}
    path=tmp_path/'plan.json';path.write_text(json.dumps(record))
    return snap,path,snapshot,record


class Server:
    def __init__(self):self.state=scene();self.requests=[]
    def handle(self,request):
        assert str(request.url).startswith('http://127.0.0.1:8765/')
        self.requests.append((request.method,request.url.path))
        if request.method=='POST':
            cmd=json.loads(request.content);action=cmd['action']
            assert request.url.path=='/api/control'
            if action=='resume':self.state['paused']=False
            elif action=='stop':self.state['paused']=True;self.state['busy']=False
            elif action in ('xyz','open','close'):
                self.state['metrics']['actions']+=1
                if action=='xyz':self.state['ee']=cmd['values']
                else:self.state['gripper_width']=.08 if action=='open' else .05
            else:raise AssertionError(action)
        return httpx.Response(200,json=copy.deepcopy(self.state))


def test_checkpoint_requires_exact_snapshot_hash_and_original_failed_open(tmp_path):
    snap,path,_,record=fixture_files(tmp_path)
    _,result,index=load_checkpoint(snap,path)
    assert index==1 and result['source_call_id']=='test-live-plan'
    snap.write_text(snap.read_text()+' ')
    with pytest.raises(ValueError,match='provenance'):load_checkpoint(snap,path)


def test_no_reset_and_only_fixed_bounded_actuator_requests():
    server=Server();backend=HTTPArmBackend(scene(),transport=httpx.MockTransport(server.handle))
    backend.paused=False;backend.manual('open');backend.manual('xyz',[.48,.21,.07]);backend.stop()
    assert backend.expected_actions==61
    assert all(path in ('/api/state','/api/control') for _,path in server.requests)
    assert server.state['ee']==[.48,.21,.07]
    with pytest.raises(ValueError):backend.manual('reset')


def test_rejects_changed_live_checkpoint_and_large_moves():
    server=Server();server.state['blocks']['blue'][0]+=.01
    with pytest.raises(ValueError,match='scene changed'):
        HTTPArmBackend(scene(),transport=httpx.MockTransport(server.handle))
    server=Server();backend=HTTPArmBackend(scene(),transport=httpx.MockTransport(server.handle));backend.paused=False
    with pytest.raises(ValueError,match='two centimetres'):backend.manual('xyz',[.48,.21,.15])
    assert server.state['metrics']['actions']==59


def test_stop_or_reset_during_pending_decision_discards_reply():
    server=Server();backend=HTTPArmBackend(scene(),transport=httpx.MockTransport(server.handle));backend.paused=False
    class Adapter:
        def decide(self,obs):
            server.state['paused']=True
            return Decision('open',1,{'open':1},True,'accepted','open')
    controller=GuardedController(Adapter(),backend,time.monotonic()+10)
    obs={'end_effector':scene()['ee'],'gripper_width':.05,'objects':scene()['blocks']}
    with pytest.raises(RuntimeError):controller.decide(obs)
    assert backend.cancelled and server.state['metrics']['actions']==59


def test_external_control_or_reset_fails_closed():
    server=Server();backend=HTTPArmBackend(scene(),transport=httpx.MockTransport(server.handle));backend.paused=False
    server.state['metrics']['actions']=60
    with pytest.raises(RuntimeError):backend.state()
    assert backend.cancelled


def test_controller_has_one_shared_call_and_deadline_budget():
    class Backend:
        def check_decision_fresh(self,obs):pass
    class Adapter:
        def decide(self,obs):return Decision('hold',1,{'hold':1},True,'accepted','hold')
    controller=GuardedController(Adapter(),Backend(),time.monotonic()+10)
    controller.calls=130
    with pytest.raises(RuntimeError):controller.decide({})
    assert controller.calls==130


def test_execute_only_failed_primitive_suffix_then_original_steps(tmp_path):
    from embodied_arm.legacy.panda.continue_stack import execute
    snap,path,snapshot,record=fixture_files(tmp_path)
    class Backend:
        cancelled=False
        def __init__(self):self.data=copy.deepcopy(snapshot);self.stops=0
        def state(self):return copy.deepcopy(self.data)
        def stop(self):self.stops+=1
    backend=Backend();seen=[]
    class Controller:
        calls=0;decisions=[]
        def cancel(self):pass
    class Leaf:
        def __init__(self,backend,controller):self.backend=backend
        def start(self,plan,**kwargs):
            seen.append(plan)
            assert plan=={'steps':[{'action':'open'},{'action':'lift','height_m':.27}]}
            self.backend.data.update(ee=[.48,.21,.27],gripper_width=.08)
            self.backend.data['blocks']['red']=[.48,.21,.025]
        def status(self):return {'status':'completed'}
        def stop(self):pass
    class Sequence:
        def __init__(self,backend,controller,allow_replan):assert allow_replan is False
        def start(self,plan):self.plan=plan;seen.append(plan)
        def status(self):return {'status':'completed','steps':[{**s,'outcome':{'verified':True}} for s in self.plan['steps']]}
        def stop(self):pass
    result=execute(snapshot,record,1,backend,Controller(),deadline=time.monotonic()+10,
                   audit_path=tmp_path/'audit.json',leaf_factory=Leaf,sequence_factory=Sequence)
    assert result['verified'] and result['new_qwen_calls']==0
    assert result['completed_prefix']==['s1']
    assert [(s['action'],s['object_id'],s['target_id']) for s in seen[1]['steps']]==[('pick','blue',None),('place','blue','on:red')]
    assert [s['source_step_id'] for s in result['remaining_step_results']]==['s2','s3','s4']


def test_failed_release_measurement_never_starts_remaining_plan(tmp_path):
    from embodied_arm.legacy.panda.continue_stack import execute
    _,_,snapshot,record=fixture_files(tmp_path)
    class Backend:
        cancelled=False
        def state(self):return copy.deepcopy(snapshot)
        def stop(self):pass
    class Controller:
        calls=0;decisions=[]
        def cancel(self):pass
    class Leaf:
        def __init__(self,*a,**kw):pass
        def start(self,*a,**kw):pass
        def status(self):return {'status':'completed'}
        def stop(self):pass
    def forbidden(*a,**kw):raise AssertionError('Unverified placement must stop')
    result=execute(snapshot,record,1,Backend(),Controller(),deadline=time.monotonic()+10,
                   audit_path=tmp_path/'audit.json',leaf_factory=Leaf,sequence_factory=forbidden)
    assert result['status']=='stopped' and not result['verified']
