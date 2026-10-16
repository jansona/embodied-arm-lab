"""UI boundary tests: no model calls, oracle data, or legacy fallback."""
from fastapi.testclient import TestClient
from embodied_arm.web.app_so101 import create_app

class SafeSimulation:
    def __init__(self,seed=0):self.paused=True;self.epoch=0;self.captures=0;self.closed=False
    def proprioception(self):return {'name':'SO-ARM101','arm_dof':5,'joints':[0]*5,'ee':[.1,0,.2],
        'gripper_openness':1,'gripper_width_estimate':.04,'busy':False,'paused':self.paused,
        'epoch':self.epoch,'actions':0,'limits':[]}
    def capture_frames(self):
        self.captures+=1
        return {'frame_id':str(self.captures),'captured_at':1.,'images':[
            {'camera_id':c,'width':640,'height':480,'data_url':'data:image/jpeg;base64,/9j/2Q=='} for c in ('left','right')],
            'calibrations':[{'camera_id':'left','intrinsics':[]}],'depth':'forbidden','segmentation':'forbidden'}
    def evaluation_state(self):raise AssertionError('UI must not access ground truth')
    def stop(self):self.paused=True
    def reset(self,seed):self.epoch+=1
    def close(self):self.closed=True

def test_safe_proprioception_and_explicit_unconfigured_vision():
    with TestClient(create_app(SafeSimulation)) as c:
        s=c.get('/api/state').json()
        assert s['robot']['name']=='SO-ARM101'
        assert s['perception']['scene'] is None and not s['perception']['configured']
        assert s['perception']['uses_ground_truth'] is False
        assert s['brain']['calls']==0 and not s['task']['available']
        assert 'blocks' not in s and 'goal' not in s

def test_stereo_preview_is_only_rgb_and_cached():
    with TestClient(create_app(SafeSimulation)) as c:
        a=c.get('/api/frames').json();b=c.get('/api/frames').json()
        assert set(a)=={'frame_id','captured_at','images'} and a==b
        assert [i['camera_id'] for i in a['images']]==['left','right']
        assert 'depth' not in a and 'segmentation' not in a

def test_reset_and_stop_stay_paused_and_reject_unknown_commands():
    with TestClient(create_app(SafeSimulation)) as c:
        c.get('/api/frames')
        assert c.post('/api/control',json={'action':'reset','seed':3}).json()['robot']['paused']
        assert c.get('/api/frames').json()['frame_id']=='2'
        assert c.post('/api/control',json={'action':'stop'}).status_code==200
        assert c.post('/api/control',json={'action':'resume'}).status_code==422
        assert c.post('/api/control',json={'action':'reset','api_key':'never'}).status_code==422
        assert c.post('/api/control',json={'action':'reset','seed':-1}).status_code==409


def test_vision_unavailable_rejects_without_starting_request():
    class Vision:
        calls=0
        def status(self):return {'available':False}
        def perceive(self,packet):self.calls+=1;raise AssertionError('Must not call')
    vision=Vision()
    with TestClient(create_app(SafeSimulation,vision=vision)) as c:
        assert c.post('/api/observe',json={}).status_code==409
        assert vision.calls==0


def test_pending_observation_duplicate_and_stop_discard_stale_result():
    import threading,time
    gate=threading.Event()
    class Vision:
        def status(self):return {'available':True,'provenance':'offline_fixture'}
    class Source:
        def __init__(self,sim,vision):self.sim=sim;self.cancelled=False;self.scene=None
        def status(self):return {'pending':not gate.is_set(),'scene':self.scene,'reason':'observing'}
        def refresh(self):gate.wait(2)
        def cancel(self):self.cancelled=True
        def snapshot(self):raise ValueError('No scene')
    with TestClient(create_app(SafeSimulation,Vision(),source_factory=Source)) as c:
        assert c.post('/api/observe',json={}).status_code==200
        assert c.post('/api/observe',json={}).status_code==409
        assert c.post('/api/control',json={'action':'stop'}).status_code==200
        gate.set();time.sleep(.03)
        state=c.get('/api/state').json()
        assert state['task']['status']=='stopped' and state['task']['plan'] is None
        assert state['robot']['paused']


def test_fresh_camera_only_plan_preview_does_not_enable_motion():
    import time
    from embodied_arm.providers.brain import BrainResult,BrainPlan
    class Vision:
        def status(self):return {'available':True,'provenance':'offline_fixture'}
    class Source:
        def __init__(self,sim,vision):self.scene=None
        def status(self):return {'scene':self.scene,'reason':'estimated','max_position_std_m':.003}
        def refresh(self):self.scene={'captured_at':time.monotonic(),'objects':{}}
        def cancel(self):self.scene=None
        def snapshot(self):
            if self.scene is None:raise ValueError('No scene')
            return {'source':'rgb_only'}
        def brain_scene(self):self.snapshot();return {'perception_source':'rgb_only'}
    class Brain:
        calls=0
        def status(self):return {'configured':True,'model':'offline_fixture','calls':self.calls}
        def interpret(self,text,scene):
            assert scene=={'perception_source':'rgb_only'}
            self.calls+=1
            return BrainResult(plan=BrainPlan('pick','red',None,'测试预览'),accepted=True,provenance='mock')
    brain=Brain()
    with TestClient(create_app(SafeSimulation,Vision(),brain,Source)) as c:
        assert c.post('/api/plan',json={'text':'夹起红色方块'}).status_code==409
        c.post('/api/observe',json={});time.sleep(.02)
        assert c.post('/api/plan',json={'text':'夹起红色方块'}).status_code==200
        time.sleep(.02);state=c.get('/api/state').json()
        assert state['task']['status']=='preview' and state['task']['provenance']=='mock'
        assert state['task']['motion_enabled'] is False and state['robot']['actions']==0
        assert brain.calls==1


def test_waypoint_default_off_and_browser_cannot_supply_coordinates():
    with TestClient(create_app(SafeSimulation)) as c:
        assert c.post('/api/waypoint',json={'object_id':'red'}).status_code==409
        assert c.post('/api/waypoint',json={'object_id':'red','offset_xyz':[1,2,3]}).status_code==422
        assert c.post('/api/waypoint',json={'object_id':'goal'}).status_code==422
        assert not c.get('/api/state').json()['waypoint']['motion_enabled']


def test_waypoint_fixed_offset_budget_and_worker_failfast():
    import time
    class Adapter:
        ready=True
        def status(self):return {'available':self.ready}
        def cancel(self):pass
    adapter=Adapter();seen=[]
    class Executor:
        def __init__(self,*args):pass
        def status(self):return {'status':'completed'}
        def run(self,obj,offset,**budgets):seen.append((obj,offset,budgets));return {'status':'completed'}
        def stop(self):pass
    with TestClient(create_app(SafeSimulation,vision=adapter,controller=adapter,enable_waypoint=True,executor_factory=Executor)) as c:
        assert c.post('/api/waypoint',json={'object_id':'blue'}).status_code==200
        time.sleep(.02)
        assert seen==[('blue',[0,0,.035],{'max_calls':12,'max_seconds':90})]
        adapter.ready=False
        assert c.post('/api/waypoint',json={'object_id':'red'}).status_code==409
        assert len(seen)==1


def test_default_camera_source_explicitly_enables_hybrid_rgb_refinement():
    with TestClient(create_app(SafeSimulation)) as c:
        s=c.get('/api/state').json()
        assert s['perception']['rgb_refinement_enabled'] is True
        assert s['perception']['provenance']=='none'
        assert not s['waypoint']['motion_enabled']


def test_stale_estimates_remain_read_only_without_planning_eligibility():
    import time
    class Source:
        def __init__(self,sim,vision):pass
        def status(self):return {'scene':{'captured_at':time.monotonic()-40,'objects':{'red':{'xyz':[.1,0,.02]}}},'reason':'estimated'}
        def snapshot(self):raise ValueError('stale')
        def cancel(self):pass
    class Brain:
        calls=0
        def status(self):return {'configured':True}
        def interpret(self,*args):self.calls+=1;raise AssertionError('Stale estimate must not be used')
    brain=Brain()
    with TestClient(create_app(SafeSimulation,brain=brain,source_factory=Source)) as c:
        s=c.get('/api/state').json()
        assert s['perception']['scene']['objects']['red']['xyz']==[.1,0,.02]
        assert s['perception']['preview_only'] and not s['perception']['fresh']
        assert s['perception']['status']=='stale_or_invalidated' and not s['task']['available']
        assert c.post('/api/plan',json={'text':'夹起红色方块'}).status_code==409 and brain.calls==0


def test_stop_discards_pending_waypoint_completion():
    import threading,time
    gate=threading.Event()
    class Adapter:
        def status(self):return {'available':True}
        def cancel(self):pass
    class Executor:
        def __init__(self,*args):self.stopped=False
        def status(self):return {'status':'stopped' if self.stopped else 'running'}
        def run(self,*args,**kwargs):gate.wait(2);return {'status':'completed'}
        def stop(self):self.stopped=True
    adapter=Adapter()
    with TestClient(create_app(SafeSimulation,vision=adapter,controller=adapter,enable_waypoint=True,executor_factory=Executor)) as c:
        assert c.post('/api/waypoint',json={'object_id':'red'}).status_code==200
        assert c.post('/api/waypoint',json={'object_id':'blue'}).status_code==409
        c.post('/api/control',json={'action':'stop'});gate.set();time.sleep(.03)
        s=c.get('/api/state').json()
        assert s['task']['status']=='stopped' and s['waypoint']['status']=='stopped'


def test_chat_works_without_scene_and_passes_history_without_motion():
    import time
    from embodied_arm.providers.brain import BrainResult,BrainPlan
    class Brain:
        def __init__(self):self.requests=[]
        def status(self):return {'available':True,'model':'offline-test'}
        def interpret(self,text,scene,history=None):
            self.requests.append((text,scene,history))
            return BrainResult(plan=BrainPlan(action='clarify',message='可以先告诉我你的需求。当前没有视觉观察。'),accepted=True,provenance='mock')
    brain=Brain()
    with TestClient(create_app(SafeSimulation,brain=brain)) as c:
        assert c.get('/api/state').json()['chat']['available']
        assert not c.get('/api/state').json()['task']['available']
        assert c.post('/api/chat',json={'text':'你好'}).status_code==200
        for _ in range(100):
            s=c.get('/api/state').json()
            if s['chat']['status']=='ready':break
            time.sleep(.01)
        assert s['chat']['status']=='ready' and len(s['chat']['messages'])==2
        assert s['robot']['actions']==0 and not s['chat']['motion_enabled']
        assert brain.requests[0][1]=={'observation_status':'unavailable','blocks':{}}
        assert c.post('/api/chat',json={'text':'我想夹红色方块'}).status_code==200
        for _ in range(100):
            if len(brain.requests)==2:break
            time.sleep(.01)
        assert [m['role'] for m in brain.requests[1][2]]==['user','assistant']
        assert c.post('/api/plan',json={'text':'夹红色方块'}).status_code==409


def test_chat_pending_cancel_and_expired_worker_are_safe():
    import threading,time
    from embodied_arm.providers.brain import BrainResult,BrainPlan
    gate=threading.Event()
    class Brain:
        ready=True;calls=0
        def status(self):return {'available':self.ready}
        def interpret(self,*args,**kwargs):
            self.calls+=1;gate.wait(2)
            return BrainResult(plan=BrainPlan(action='status',message='迟到回复'),accepted=True)
        def cancel(self):pass
    brain=Brain()
    with TestClient(create_app(SafeSimulation,brain=brain)) as c:
        c.post('/api/chat',json={'text':'状态'})
        assert c.post('/api/chat',json={'text':'重复'}).status_code==409
        c.post('/api/control',json={'action':'stop'});gate.set();time.sleep(.03)
        s=c.get('/api/state').json()
        assert s['chat']['status']=='cancelled'
        assert [m['role'] for m in s['chat']['messages']]==['user']
        brain.ready=False
        assert c.post('/api/chat',json={'text':'再来'}).status_code==409
        assert brain.calls==1 and s['robot']['actions']==0


def test_chat_rejects_actionable_reply_and_renders_no_motion():
    import time
    from embodied_arm.providers.brain import BrainResult,BrainPlan
    class Brain:
        def status(self):return {'available':True}
        def interpret(self,*args,**kwargs):return BrainResult(plan=BrainPlan(action='pick',object_id='red',message='执行'),accepted=True)
    with TestClient(create_app(SafeSimulation,brain=Brain())) as c:
        c.post('/api/chat',json={'text':'抓红色'})
        for _ in range(100):
            s=c.get('/api/state').json()
            if s['chat']['status']=='error':break
            time.sleep(.01)
        assert s['chat']['status']=='error' and len(s['chat']['messages'])==1
        assert s['robot']['actions']==0 and s['task']['plan'] is None


def test_one_text_task_observes_plans_executes_and_reports_verified_outcome():
    import time
    from embodied_arm.providers.brain import BrainResult,BrainPlan
    trace=[]
    class Source:
        def __init__(self,sim,vision):pass
        def status(self):return {'scene':None,'reason':'none'}
        def refresh(self):trace.append('observe')
        def brain_scene(self):return {'perception_source':'rgb_only'}
        def snapshot(self):trace.append('fresh')
        def cancel(self):pass
    class Brain:
        def status(self):return {'available':True}
        def interpret(self,text,scene,history):
            assert scene=={'perception_source':'rgb_only'};trace.append('plan')
            return BrainResult(plan=BrainPlan(action='pick',object_id='red',message='测试动作计划'),accepted=True,provenance='mock')
    class Executor:
        validated=True
        def execute(self,plan,*,cancel_event,event_callback):
            trace.append('execute');event_callback({'phase':'lift','provenance':'mock_fixture'})
            return {'verified':True,'source':'mock_fixture'}
        def stop(self):pass
    with TestClient(create_app(SafeSimulation,brain=Brain(),source_factory=Source,task_executor=Executor())) as c:
        assert c.post('/api/tasks',json={'text':'夹起红色方块','execute':True}).status_code==200
        for _ in range(100):
            state=c.get('/api/state').json()
            if state['workflow']['state']=='completed':break
            time.sleep(.01)
        assert trace==['observe','plan','fresh','execute']
        assert state['workflow']['outcome']['verified']
        assert [e['state'] for e in state['workflow']['events']]==['observing','planning','executing','executing','completed']
        assert state['workflow']['provenance']=='mock'


def test_unvalidated_task_execution_is_unavailable_without_model_call():
    class Brain:
        def status(self):return {'available':True}
        def interpret(self,*args):raise AssertionError('No call')
    with TestClient(create_app(SafeSimulation,brain=Brain())) as c:
        assert c.post('/api/tasks',json={'text':'夹起红色方块','execute':True}).status_code==409
        assert not c.get('/api/state').json()['workflow']['execution_available']


def test_reset_clears_executor_held_state_but_stop_preserves_it():
    class Executor:
        validated=True
        held='red';resets=0;stops=0
        def stop(self):self.stops+=1
        def reset(self):self.held=None;self.resets+=1
    executor=Executor()
    with TestClient(create_app(SafeSimulation,task_executor=executor)) as c:
        assert c.post('/api/control',json={'action':'reset','seed':-1}).status_code==409
        assert executor.held=='red' and executor.stops==0 and executor.resets==0
        c.post('/api/control',json={'action':'stop'})
        assert executor.held=='red' and executor.resets==0
        c.post('/api/control',json={'action':'reset','seed':2})
        assert executor.held is None and executor.resets==1
        s=c.get('/api/state').json()
        assert s['chat']['messages']==[] and s['robot']['paused']


def test_task_executor_factory_receives_the_same_safe_runtime_dependencies():
    seen=[];controller=object()
    class Executor:
        validated=False
        def stop(self):pass
    def factory(sim,scene_source,adapter):
        seen.append((sim,scene_source,adapter));return Executor()
    app=create_app(SafeSimulation,controller=controller,task_executor_factory=factory)
    with TestClient(app) as c:
        session=app.state.session
        assert seen==[(session.sim,session.source,controller)]
        assert not c.get('/api/state').json()['workflow']['execution_available']
