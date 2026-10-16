"""Offline API integration. No saved credentials or network calls are used."""
import threading
import time
from fastapi.testclient import TestClient
import app as module
from brain import BrainPlan, BrainResult


class FakeBrain:
    def __init__(self, action='pick', gate=None):self.action=action;self.gate=gate;self.calls=0
    def status(self):return {'configured':True,'model':'mock/test','calls':self.calls}
    def interpret(self,text,scene,history=None):
        self.calls+=1
        if self.gate:self.gate.wait(3)
        return BrainResult(plan=BrainPlan(action=self.action,object_id='red' if self.action=='pick' else None,message='已理解测试请求'),accepted=True)


def settle(client, predicate, limit=3):
    until=time.monotonic()+limit
    while time.monotonic()<until:
        state=client.get('/api/state').json()
        if predicate(state['chat']):return state
        time.sleep(.01)
    raise AssertionError(state['chat'])


def test_chat_pick_roundtrip_and_stop():
    with TestClient(module.app) as c:
        module.brain=FakeBrain()
        r=c.post('/api/chat',json={'text':'夹起这个红色方块','controller':'deterministic'})
        assert r.status_code==200
        s=settle(c,lambda x:x['status']=='executing')
        assert s['chat']['plan']['object_id']=='red'
        assert s['chat']['messages'][0]['content']=='夹起这个红色方块'
        assert s['chat']['task']['controller']=='deterministic servo'
        assert c.post('/api/chat',json={'text':'再来一次'}).status_code==409
        assert c.post('/api/control',json={'action':'open'}).status_code==409
        assert c.post('/api/chat/stop').json()['chat']['status']=='stopped'
        assert module.brain.calls==1


def test_stop_discards_late_brain_result_and_repeated_send():
    gate=threading.Event()
    with TestClient(module.app) as c:
        module.brain=FakeBrain(gate=gate)
        assert c.post('/api/chat',json={'text':'夹起红色方块'}).status_code==200
        assert c.post('/api/chat',json={'text':'夹起蓝色方块'}).status_code==409
        stopped=c.post('/api/chat/stop').json()['chat']
        assert stopped['status']=='stopped' and stopped['request_pending']
        assert c.post('/api/chat',json={'text':'再来'}).status_code==409
        gate.set()
        s=settle(c,lambda x:not x['request_pending'])
        assert s['chat']['plan'] is None and s['chat']['task']['actions']==0
        assert s['chat']['status']=='stopped'


def test_reset_cancels_pending_and_no_controller_fallback():
    gate=threading.Event()
    with TestClient(module.app) as c:
        module.brain=FakeBrain(gate=gate)
        assert c.post('/api/chat',json={'text':'红色','controller':'jev'}).status_code==409
        assert module.brain.calls==0
        assert c.post('/api/chat',json={'text':'夹起红色方块'}).status_code==200
        assert c.post('/api/control',json={'action':'reset'}).status_code==200
        gate.set();s=settle(c,lambda x:not x['request_pending'])
        assert s['chat']['task']['actions']==0 and s['chat']['plan'] is None


def test_clarify_never_actuates_and_input_validation():
    with TestClient(module.app) as c:
        module.brain=FakeBrain(action='clarify')
        assert c.post('/api/chat',json={'text':'夹起那个'}).status_code==200
        s=settle(c,lambda x:x['status']=='clarify')
        assert s['chat']['task']['actions']==0
        assert c.post('/api/chat',json={'text':' '}).status_code==422
        assert c.post('/api/chat',json={'text':'你好','api_key':'forbidden'}).status_code==422
        assert c.post('/api/chat',json={'text':'你好'},headers={'Origin':'https://evil.example'}).status_code==403


def test_backend_errors_do_not_expose_provider_contents():
    class Broken(FakeBrain):
        def interpret(self,*args,**kwargs):raise RuntimeError('secret-key-provider-body')
    with TestClient(module.app) as c:
        module.brain=Broken()
        c.post('/api/chat',json={'text':'夹起红色方块'})
        s=settle(c,lambda x:x['status']=='error')
        assert 'secret-key' not in str(s)
        assert s['chat']['task']['actions']==0


def test_explicit_offline_configuration_uses_real_mock_and_no_config(monkeypatch):
    def denied(*args,**kwargs):raise AssertionError('Offline mode must not load credentials')
    monkeypatch.setattr(module.JevRunner,'configure',denied)
    with TestClient(module.app) as c:
        s=c.post('/api/brain/configure',json={'mode':'mock'}).json()
        assert s['chat']['brain']['configured']
        assert s['chat']['brain']['model']=='offline-fixed-examples'
        assert c.post('/api/chat',json={'text':'夹起这个红色方块'}).status_code==200
        s=settle(c,lambda x:x['status']=='executing')
        assert s['chat']['provenance']=='mock'
        assert s['chat']['brain']['calls']==0
        assert s['chat']['plan']['action']=='pick'
        assert s['chat']['plan']['object_id']=='red'
        c.post('/api/chat/stop')


def test_motion_completion_does_not_claim_success_for_missed_grasp():
    with TestClient(module.app) as c:
        module.chat_data.update(status='executing',plan={'action':'pick','object_id':'red','target_id':None,'message':'测试'})
        module.executor.result['status']='completed'
        s=c.get('/api/state').json()['chat']
        assert s['status']=='completed'
        assert s['outcome']['verified'] is False
        assert '未达到' in s['outcome']['description']


def test_provider_error_preserves_call_count_after_worker_exit():
    class HttpFailure(FakeBrain):
        def interpret(self,*args,**kwargs):return BrainResult(reason='http_error_400',calls=1)
    with TestClient(module.app) as c:
        module.brain=HttpFailure()
        c.post('/api/chat',json={'text':'夹起红色方块'})
        s=settle(c,lambda x:x['status']=='error')['chat']
        assert s['brain']['calls']==1
        assert s['error_code']=='http_error_400'
        assert s['task']['actions']==0


def test_sequence_chat_routes_model_order_and_explicit_repair_flag(monkeypatch):
    from brain import BrainStep
    captured=[]
    class SequenceBrain(FakeBrain):
        def interpret(self,*args,**kwargs):
            return BrainResult(accepted=True,plan=BrainPlan(action='sequence',message='蓝色叠到红色',
                steps=(BrainStep('s1','pick','blue'),BrainStep('s2','place','blue','on:red'))))
    class SequenceFake:
        def __init__(self,*args,**kwargs):captured.append(kwargs);self.result={'status':'idle','actions':0}
        def start(self,plan):captured.append(plan);self.result={'status':'running','actions':0,'steps':plan['steps']}
        def status(self):return dict(self.result)
        def stop(self):self.result['status']='stopped'
    monkeypatch.setattr(module,'SequenceExecutor',SequenceFake)
    with TestClient(module.app) as c:
        module.brain=SequenceBrain()
        assert c.post('/api/chat',json={'text':'把两个方块叠起来','allow_replan':True}).status_code==200
        s=settle(c,lambda x:x['status']=='executing')['chat']
        assert s['plan']['action']=='sequence'
        assert [x['action'] for x in captured[1]['steps']]==['pick','place']
        assert captured[0]['allow_replan'] is True
        assert captured[0]['controller'] is None
        assert c.post('/api/chat',json={'text':'再来'}).status_code==409
        assert c.post('/api/chat/stop').json()['chat']['status']=='stopped'


def test_recovery_is_one_use_jev_only_without_brain_calls(monkeypatch,tmp_path):
    import json
    path=tmp_path/'approved.json'
    path.write_text(json.dumps({'plan':{'action':'pick','object_id':'red','target_id':None,'message':'已验证计划'},
        'source_call_id':'offline-test-source','model':'qwen3.8-max','provenance':'live','accepted':True}))
    monkeypatch.setattr(module,'RECOVERY_PATH',path)
    created=[]
    class RecoveryTask:
        def __init__(self,sim,controller=None):self.result={'status':'idle','actions':0,'controller_calls':0};created.append(controller)
        def start(self,plan):self.result['status']='running'
        def status(self):return dict(self.result)
        def stop(self):self.result['status']='stopped'
    with TestClient(module.app) as c:
        adapter=object();module.runner._adapter=adapter;module.brain=FakeBrain()
        monkeypatch.setattr(module,'TaskExecutor',RecoveryTask)
        assert c.get('/api/state').json()['recovery']['available']
        s=c.post('/api/chat/recover').json()
        assert s['chat']['provenance']=='reused_live'
        assert s['chat']['controller']=='jev' and created==[adapter]
        assert module.brain.calls==0
        assert not s['recovery']['available'] and path.with_suffix('.consumed.json').exists()
        assert c.post('/api/chat/recover').status_code==409
        c.post('/api/chat/stop')


def test_recovery_rejects_unapproved_object_and_cannot_fallback(monkeypatch,tmp_path):
    import json
    path=tmp_path/'approved.json';monkeypatch.setattr(module,'RECOVERY_PATH',path)
    path.write_text(json.dumps({'plan':{'action':'pick','object_id':'blue','target_id':None,'message':'蓝色'},
        'source_call_id':'test','model':'qwen3.8-max','provenance':'live','accepted':True}))
    with TestClient(module.app) as c:
        assert not c.get('/api/state').json()['recovery']['available']
        assert c.post('/api/chat/recover').status_code==409
        assert module.executor.status()['actions']==0


def test_exact_sequence_recovery_uses_explicit_larger_bounded_preset(monkeypatch,tmp_path):
    import json
    steps=[{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
           {'step_id':'s2','action':'place','object_id':'red','target_id':'goal'},
           {'step_id':'s3','action':'pick','object_id':'blue','target_id':None},
           {'step_id':'s4','action':'place','object_id':'blue','target_id':'on:red'}]
    plan={'action':'sequence','object_id':None,'target_id':None,'message':'原始四步','steps':steps}
    path=tmp_path/'approved.json';path.write_text(json.dumps({'plan':plan,'source_call_id':'test-original',
        'model':'qwen3.8-max','provenance':'live','accepted':True}))
    monkeypatch.setattr(module,'RECOVERY_PATH',path);captured=[]
    class SequenceFake:
        def __init__(self,sim,**kwargs):captured.append(kwargs);self.result={'status':'idle','actions':0}
        def start(self,p):captured.append(p);self.result['status']='running'
        def status(self):return dict(self.result)
        def stop(self):self.result['status']='stopped'
    with TestClient(module.app) as c:
        module.runner._adapter=object();module.brain=FakeBrain()
        monkeypatch.setattr(module,'SequenceExecutor',SequenceFake)
        s=c.post('/api/chat/recover').json()
        assert s['chat']['provenance']=='reused_live' and module.brain.calls==0
        assert captured[0]['max_controller_calls']==200 and captured[0]['max_seconds']==420
        assert captured[0]['allow_replan'] is False and captured[1]==plan
        assert s['seed']==0
        assert c.post('/api/chat/recover').status_code==409
        c.post('/api/chat/stop')


def test_expired_jev_worker_rejects_before_any_brain_call(monkeypatch):
    class OfflineAdapter:
        def status(self):return {'available':False}
        def cancel(self):pass
    with TestClient(module.app) as c:
        module.brain=FakeBrain();module.runner._adapter=OfflineAdapter()
        monkeypatch.setattr(module,'IPC_MODE',True)
        response=c.post('/api/chat',json={'text':'把两个方块垒起来','controller':'jev'})
        assert response.status_code==409
        assert 'no brain request' in response.json()['detail']
        assert module.brain.calls==0
        assert module.executor.status()['actions']==0
