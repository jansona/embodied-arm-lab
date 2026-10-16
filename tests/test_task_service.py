import threading,time
from types import SimpleNamespace
import pytest
from embodied_arm.services.task_service import TaskService, UNKNOWN_SCENE
class Source:
 def refresh(self):pass
 def brain_scene(self):return {'blocks':{'red':[.2,0,.01]},'goal':[.25,0,.01],'ee':[.2,0,.06]}
 def snapshot(self):return {}
 def cancel(self):pass
class Brain:
 def __init__(self,action='clarify'):self.action=action;self.seen=[]
 def status(self):return {'available':True}
 def interpret(self,text,scene,history):
  self.seen.append((scene,history))
  plan={'action':self.action,'object_id':'red' if self.action=='pick' else None,'target_id':None,'message':'回复'}
  return SimpleNamespace(accepted=True,plan=SimpleNamespace(to_dict=lambda:plan),provenance='mock')
 def cancel(self):pass
class Executor:
 validated=True
 def __init__(self):self.calls=0
 def execute(self,plan,**kwargs):self.calls+=1;return {'verified':True,'provenance':'offline_fixture'}
 def stop(self):pass
def wait(service):
 for _ in range(100):
  if not service.status()['pending']:return service.status()
  time.sleep(.01)
 raise AssertionError('did not finish')
def test_chat_without_scene_or_motion():
 b=Brain();e=Executor();s=TaskService(scene_source=Source(),brain=b,executor=e)
 s.submit('你好',execute=False);r=wait(s)
 assert r['state']=='completed' and e.calls==0 and b.seen[0][0]==UNKNOWN_SCENE
 assert r['provenance']=='mock'
def test_image_plan_execute_lifecycle():
 e=Executor();s=TaskService(scene_source=Source(),brain=Brain('pick'),executor=e)
 s.submit('拿红块');r=wait(s)
 assert [x['state'] for x in r['events']]==['observing','planning','executing','completed']
 assert e.calls==1 and r['outcome']['verified']
def test_uncertain_image_blocks_no_model_no_motion():
 class Bad(Source):
  def refresh(self):raise ValueError('private detail')
 b=Brain('pick');e=Executor();s=TaskService(scene_source=Bad(),brain=b,executor=e)
 s.submit('拿红块');r=wait(s)
 assert r['state']=='blocked' and r['error']=='observation_unavailable'
 assert b.seen==[] and e.calls==0
def test_unvalidated_executor_not_advertised():
 e=Executor();e.validated=False;s=TaskService(scene_source=Source(),brain=Brain(),executor=e)
 assert not s.status()['execution_available']
 with pytest.raises(ValueError,match='execution_unavailable'):s.submit('拿红块')
def test_stop_rejects_pending_reply_and_duplicate():
 gate=threading.Event()
 class Slow(Brain):
  def interpret(self,*args):gate.wait(2);return super().interpret(*args)
 b=Slow('pick');e=Executor();s=TaskService(scene_source=Source(),brain=b,executor=e)
 s.submit('拿红块')
 with pytest.raises(ValueError,match='task_busy'):s.submit('再拿')
 s.stop();gate.set();r=wait(s)
 assert r['state']=='cancelled' and r['plan'] is None and e.calls==0

def test_expired_jev_blocks_before_observation_or_brain():
 class Unavailable(Executor):
  def readiness(self):return {'available':False,'reason':'jev_unavailable'}
 class Untouched(Source):
  def refresh(self):raise AssertionError('Must fail before spending vision')
 b=Brain();e=Unavailable();s=TaskService(scene_source=Untouched(),brain=b,executor=e)
 assert s.status()['unavailable_reason']=='jev_unavailable'
 with pytest.raises(ValueError,match='execution_unavailable'):s.submit('拿红块')
 assert not b.seen and not e.calls
 s.submit('你好',execute=False)
 assert wait(s)['state']=='completed'
