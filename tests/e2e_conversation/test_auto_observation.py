"""HTTP lifecycle contracts with MOCK scene/provider. Not live API or physical evidence."""
import threading
import time
from contextlib import contextmanager

from fastapi.testclient import TestClient
from embodied_arm.web.app_so101 import create_app
from embodied_arm.providers.brain import BrainPlan, BrainResult
from tests.e2e_conversation.test_browser import FakeRobot


class MockVision:
    def __init__(self): self.available=True; self.calls=0
    def status(self): return {'available':self.available,'model':'MOCK-VISION','calls':self.calls}
    def cancel(self): pass


class MockSource:
    """Synthetic scene contract; no image recognition or simulator truth access."""
    def __init__(self,sim,vision=None):
        self.sim=sim; self.vision=vision; self.scene=None; self.refreshes=0
        self.gate=threading.Event(); self.gate.set(); self.cancelled=False; self.stale=False
    def status(self):
        return {'scene':self.scene,'reason':'MOCK-SCENE','max_position_std_m':.003,
                'last_vision':{'provenance':'MOCK-NOT-LIVE'}}
    def refresh(self,*args,**kwargs):
        self.refreshes+=1
        if not self.vision.available: raise ValueError('vision_unavailable')
        self.cancelled=False
        self.sim.capture_frames(); self.vision.calls+=1; self.gate.wait(5)
        if self.cancelled: raise ValueError('cancelled')
        self.stale=False; self.scene={'captured_at':time.monotonic(),'objects':{}}
        return self.snapshot()
    def ensure_brain_scene(self):
        try:return self.brain_scene()
        except ValueError:self.refresh();return self.brain_scene()
    def snapshot(self):
        if self.scene is None or self.stale: raise ValueError('stale_frame')
        return {'mock_observed':True,'source':'rgb_only'}
    def brain_scene(self):
        self.snapshot()
        return {'mock_observed':True,'perception_source':'rgb_only','blocks':{},
                'detector_provenance':'MOCK-NOT-LIVE'}
    def cancel(self): self.cancelled=True;self.scene=None


class MockBrain:
    def __init__(self): self.calls=[];self.available=True
    def status(self):return {'available':self.available,'configured':self.available,'model':'MOCK-BRAIN','calls':len(self.calls)}
    def interpret(self,text,scene,history=None):
        self.calls.append((text,scene,history))
        plan=BrainPlan('pick','red',None,'MOCK：已获取本地相机观察，仅生成计划。') if scene.get('mock_observed') else BrainPlan('clarify',message='MOCK：普通对话')
        return BrainResult(accepted=True,plan=plan,provenance='MOCK-NOT-LIVE')
    def cancel(self):pass


@contextmanager
def mock_app():
    brain=MockBrain();vision=MockVision();app=create_app(FakeRobot,vision=vision,brain=brain,source_factory=MockSource)
    with TestClient(app) as client:
        yield client,app.state.session,brain,vision


def finished(client):
    for _ in range(200):
        state=client.get('/api/state').json()
        if not state['pending']:return state
        time.sleep(.01)
    raise AssertionError('MOCK workflow did not terminate')


def test_first_red_cube_instruction_observes_without_manual_recognize():
    with mock_app() as (c,s,b,v):
        assert s.source.scene is None
        assert c.post('/api/chat',json={'text':'请夹起红色方块'}).status_code==200
        state=finished(c)
        assert s.source.refreshes==1
        assert len(b.calls)==1 and b.calls[0][1]['mock_observed'] is True
        assert state['workflow']['state']=='plan_preview'
        assert not state['workflow']['execute'] and state['robot']['actions']==0
        assert state['workflow']['provenance']=='MOCK-NOT-LIVE'
        assert [event['state'] for event in state['workflow']['events']][:2]==['observing','planning']


def test_unavailable_vision_never_falls_back_to_brain_unknown_scene():
    with mock_app() as (c,s,b,v):
        v.available=False
        c.post('/api/chat',json={'text':'红色方块在哪里？'})
        state=finished(c)
        assert state['workflow']['state']=='blocked'
        assert not b.calls
        assert state['robot']['actions']==0
        assert state['workflow'].get('error') in ('vision_unavailable','observation_unavailable','unavailable')


def test_stale_scene_is_refreshed_before_relevant_instruction():
    with mock_app() as (c,s,b,v):
        s.source.scene={'captured_at':time.monotonic()-100,'objects':{}};s.source.stale=True
        c.post('/api/chat',json={'text':'夹起红色方块'})
        state=finished(c)
        assert s.source.refreshes==1 and not s.source.stale
        assert state['workflow']['state']=='plan_preview'
        assert b.calls[0][1]['mock_observed'] is True


def test_stop_during_auto_observation_blocks_late_plan_and_motion():
    with mock_app() as (c,s,b,v):
        s.source.gate.clear()
        c.post('/api/chat',json={'text':'夹起红色方块'})
        for _ in range(100):
            if s.source.refreshes:break
            time.sleep(.01)
        assert s.source.refreshes==1
        assert c.post('/api/control',json={'action':'stop'}).status_code==200
        s.source.gate.set();state=finished(c)
        assert state['workflow']['state']=='cancelled'
        assert not b.calls and state['robot']['actions']==0


def test_reconnect_retries_observation_without_manual_click():
    with mock_app() as (c,s,b,v):
        v.available=False;c.post('/api/chat',json={'text':'夹起红色方块'});finished(c)
        v.available=True;c.post('/api/chat',json={'text':'夹起红色方块'});state=finished(c)
        assert state['workflow']['state']=='plan_preview'
        assert len(b.calls)==1 and state['robot']['actions']==0


def test_preview_frames_never_spend_provider_calls():
    with mock_app() as (c,s,b,v):
        for _ in range(100):
            if c.get('/api/frames').status_code==200:break
            time.sleep(.01)
        for _ in range(3):
            response=c.get('/api/frames');assert response.status_code==200
            assert set(response.json())=={'frame_id','captured_at','images'}
        assert not b.calls and v.calls==0 and s.source.refreshes==0


def test_cached_valid_scene_is_reused_without_new_paid_recognition():
    with mock_app() as (c,s,b,v):
        s.source.refresh()
        c.post('/api/chat',json={'text':'红色方块在哪里？'});state=finished(c)
        assert s.source.refreshes==1 and v.calls==1
        assert b.calls[0][1]['mock_observed'] is True
        assert state['robot']['actions']==0


def test_missing_brain_configuration_never_starts_recognition():
    with mock_app() as (c,s,b,v):
        b.available=False
        assert c.post('/api/chat',json={'text':'夹起红色方块'}).status_code==409
        assert not b.calls and v.calls==0 and s.source.refreshes==0


def test_continuous_local_capture_increments_frames_without_recognition():
    with mock_app() as (c,s,b,v):
        first=s.sim.captures
        deadline=time.monotonic()+2.5
        while s.sim.captures<=first and time.monotonic()<deadline:time.sleep(.02)
        state=c.get('/api/state').json()
        assert s.sim.captures>first
        assert state['camera']['interval_seconds']==1
        assert state['camera']['model_calls']==0
        assert v.calls==0 and not b.calls


def test_preview_is_explicitly_non_motion_and_stale_basis_is_visible():
    with mock_app() as (c,s,b,v):
        c.post('/api/chat',json={'text':'夹起红色方块'});state=finished(c)
        assert state['workflow']['plan_fresh'] is True
        assert state['workflow']['motion_executed'] is False
        s.source.stale=True
        assert c.get('/api/state').json()['workflow']['plan_fresh'] is False
        assert s.sim.proprioception()['actions']==0
