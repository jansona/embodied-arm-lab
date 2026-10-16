"""New generic web boundary tests; only explicit fake service/providers."""
from pathlib import Path
from fastapi.testclient import TestClient
from embodied_arm.web.app_visual import create_app

class Sim:
    def __init__(self):self.epoch=0;self.closed=False
    def stop(self):pass
    def tick(self,count):pass
    def proprioception(self):return {'name':'SO-ARM101','arm_dof':5,'epoch':self.epoch,'actions':0,'ee':[.1,0,.2],'joints':[0]*5,'joint_limits':[[-1,1]]*5,'workspace':[[0,.4],[-.2,.2],[0,.4]],'gripper_openness':1.,'paused':True,'busy':False,'cube_edge':.02}
    def capture_frames(self):return {'frame_id':'f1','captured_at':1.,'images':[{'camera_id':c,'width':640,'height':480,'data_url':'data:image/jpeg;base64,/9j/2Q=='} for c in ('left','right')],'oracle':'never'}
    def close(self):self.closed=True
    def evaluation_state(self):raise AssertionError('No oracle')
class Provider:
    def status(self):return {'available':True,'provenance':'fake_test','calls':0}
class Service:
    def __init__(self,sim,qwen,jev,**budgets):self.sim=sim;self.budgets=budgets;self.record={'state':'idle','pending':False,'events':[],'history':[],'execution_available':True};self.submits=[];self.closed=False
    def status(self):return dict(self.record)
    def submit(self,text,execute=True):
        if self.record['pending']:raise ValueError('task_busy')
        if not text.strip():raise ValueError('invalid_text')
        self.submits.append((text,execute));self.record.update(state='observing',pending=True)
    def stop(self):self.record.update(state='cancelled',pending=False)
    def reset(self):self.sim.epoch+=1;self.record.update(state='idle',pending=False)
    def close(self):self.closed=True

def test_new_app_routes_have_no_legacy_pipeline_or_oracle():
    app=create_app(Sim,Provider(),Provider(),Service)
    with TestClient(app) as c:
        assert c.get('/').status_code==200
        state=c.get('/api/state').json();assert state['schema']=='visual-action-ui-v1'
        assert 'cube_edge' not in state['robot']
        assert 'perception' not in state and 'oracle' not in c.get('/api/frames').json()
        assert c.post('/api/observe',json={}).status_code==404
        assert c.post('/api/waypoint',json={'object_id':'red'}).status_code==404
        assert app.state.service.budgets=={'cycle_max_calls':32,'max_calls':200,'max_seconds':420}
    assert app.state.service.closed and app.state.sim.closed
    source=Path('embodied_arm/web/app_visual.py').read_text()
    for old in ('CameraSceneSource','manipulation','brain_ipc','vision_ipc','app_so101'):
        assert old not in source.replace('scene/CV/manipulation pipeline','pipeline')

def test_submit_stop_then_new_command_without_reset():
    app=create_app(Sim,Provider(),Provider(),Service)
    with TestClient(app) as c:
        assert c.post('/api/tasks',json={'text':'移动到杯子旁边'}).status_code==200
        assert app.state.service.submits==[('移动到杯子旁边',True)]
        assert c.post('/api/tasks',json={'text':'另一个任务'}).status_code==409
        assert c.post('/api/control',json={'action':'reset'}).status_code==409
        assert c.post('/api/control',json={'action':'stop'}).status_code==200
        assert app.state.sim.epoch==0
        assert c.post('/api/tasks',json={'text':'查看当前状态','execute':False}).status_code==200
        assert app.state.service.submits[-1]==('查看当前状态',False)
        assert c.post('/api/tasks',json={'text':'x','api_key':'not-accepted'}).status_code==422
        assert c.post('/api/control',json={'action':'move','xyz':[1,2,3]}).status_code==422

def test_default_provider_status_is_unconfigured_without_live_import():
    app=create_app(Sim,service_factory=Service)
    with TestClient(app) as c:
        state=c.get('/api/state').json()
        assert state['qwen']['available'] is False and state['jev']['available'] is False

def test_close_defers_simulator_destruction_until_service_is_settled():
    class PendingClose(Service):
        def close(self):self.closed=True;return False
        def wait_closed(self,timeout):return False
    app=create_app(Sim,Provider(),Provider(),PendingClose,cycle_max_calls=7)
    with TestClient(app):assert app.state.service.budgets['cycle_max_calls']==7
    assert app.state.service.closed and not app.state.sim.closed


def test_deferred_close_runs_once_after_late_provider_settles():
    import threading,time
    gate=threading.Event()
    class CountSim(Sim):
        def __init__(self):super().__init__();self.closes=0
        def close(self):self.closes+=1;super().close()
    class LateService(Service):
        def close(self):return False
        def wait_closed(self,timeout):
            assert timeout==65
            return gate.wait(2)
    app=create_app(CountSim,Provider(),Provider(),LateService)
    with TestClient(app):pass
    assert app.state.sim.closes==0
    gate.set()
    deadline=time.monotonic()+2
    while not app.state.sim.closed and time.monotonic()<deadline:time.sleep(.01)
    assert app.state.sim.closed and app.state.sim.closes==1
