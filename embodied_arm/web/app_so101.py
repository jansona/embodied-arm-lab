"""SO-ARM101 stereo-image UI. No oracle scene enters the control surface.

Separate entry point from the validated Panda baseline. Provider configuration is
never loaded here, and display camera capture does not trigger model requests.
"""
from contextlib import asynccontextmanager
import threading
import os
import time
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

ROOT=Path(__file__).resolve().parent

class ControlRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action:Literal['stop','reset']
    seed:int=0

class PlanRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str

class TaskRequest(PlanRequest):
    execute:bool=True

class WaypointRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    object_id:Literal['red','blue']

class SO101Session:
    def __init__(self,sim,vision=None,brain=None,source_factory=None,controller=None,enable_waypoint=False,executor_factory=None,task_executor=None,task_executor_factory=None,camera_interval_seconds=1.):
        self.sim=sim;self.lock=threading.RLock();self.closed=False
        self.frames=None;self.frames_time=0.;self.generation=0
        from embodied_arm.services.camera_sampler import CameraSampler
        self.camera=CameraSampler(sim,interval_seconds=camera_interval_seconds)
        if source_factory is None:
            from embodied_arm.control.so101_control import CameraSceneSource
            self.source=CameraSceneSource(sim,vision=vision,refine_rgb=True)
        else:
            self.source=source_factory(sim,vision=vision)
        self.vision=vision;self.brain=brain
        self.worker=None;self.task={'status':'idle','available':False,'reason':'preview_only','plan':None}
        self.error=None;self.controller=controller;self.enable_waypoint=enable_waypoint
        self.executor_factory=executor_factory;self.waypoint_executor=None
        from embodied_arm.services.task_service import TaskService
        if task_executor is not None and task_executor_factory is not None:raise ValueError('Choose one task executor injection')
        if task_executor_factory is not None:task_executor=task_executor_factory(sim,self.source,controller)
        self.task_service=TaskService(scene_source=self.source,brain=brain,executor=task_executor)
    def _pending(self):
        return bool((self.worker and self.worker.is_alive()) or self.task_service.status()['pending'])
    def _controller_status(self):
        return self.controller.status() if self.controller and hasattr(self.controller,'status') else {'available':False}
    def state(self):
        with self.lock:
            vision=self.vision.status() if self.vision and hasattr(self.vision,'status') else {'available':False}
            perception=self.source.status()
            fresh=False
            if perception.get('scene') is not None:
                try:self.source.snapshot();fresh=True
                except ValueError:perception={**perception,'reason':'stale_or_invalidated'}
            brain_status=self.brain.status() if self.brain else {'configured':False,'model':'qwen3.8-max','calls':0}
            brain_ready=bool(brain_status.get('available',brain_status.get('configured',False)))
            workflow=self.task_service.status()
            chat_state={'queued':'thinking','planning':'thinking','completed':'ready','failed':'error','blocked':'error'}.get(workflow['state'],workflow['state'])
            return {'schema':'so101-ui-v1','robot':self.sim.proprioception(),'camera':self.camera.status(),
                'perception':{**perception,'configured':bool(vision.get('available',vision.get('configured',False))),
                    'status':'ready' if fresh else perception.get('reason','unconfigured'),
                    'model':vision.get('model','qwen3.8-max'),'provenance':(perception.get('last_vision') or {}).get('provenance','none'),
                    'image_source':'simulated_rgb_stereo','uses_ground_truth':False,
                    'worker':vision,'fresh':fresh,'preview_only':bool(perception.get('scene') and not fresh),
                    'age_seconds':max(0,time.monotonic()-perception['scene']['captured_at']) if perception.get('scene') else None},
                'brain':brain_status,
                'chat':{'available':brain_ready,'status':chat_state,'messages':workflow['history'],
                        'error':workflow.get('error'),'provenance':workflow.get('provenance','none'),'motion_enabled':False},
                'workflow':workflow,
                'controller':{**self._controller_status(),'model':'decision-model-preview','fallback':False},
                'task':{**self.task,'available':bool(brain_ready and fresh),'motion_enabled':False,'scene_fresh':fresh},
                'waypoint':{**(self.waypoint_executor.status() if self.waypoint_executor else {'status':'idle'}),
                    'available':bool(self.enable_waypoint and vision.get('available',False) and self._controller_status().get('available',False)),
                    'motion_enabled':self.enable_waypoint,'max_calls':12,'max_seconds':90,'offset_xyz':[0,0,.035]},
                'error':self.error,'pending':self._pending(),
                'evaluation':{'isolated':True,'exposed_to_ai':False,'visible':False}}
    def observe(self):
        with self.lock:
            if self._pending():raise ValueError('A request is already pending')
            status=self.vision.status() if self.vision and hasattr(self.vision,'status') else {}
            if not status.get('available',status.get('configured',False)):raise ValueError('Vision worker unavailable; no request sent')
            self.generation+=1;epoch=self.generation;self.error=None
            self.task={'status':'observing','available':False,'reason':'preview_only','plan':None}
            self.worker=threading.Thread(target=self._observe,args=(epoch,),daemon=True);self.worker.start()
            return self.state()
    def _observe(self,epoch):
        try:self.source.refresh();reason=None
        except Exception:reason='observation_rejected_or_unavailable'
        with self.lock:
            if epoch!=self.generation:return
            self.error=reason;self.task['status']='observation_failed' if reason else 'observed'
    def submit_task(self,text,execute=True):
        with self.lock:
            if self._pending():raise ValueError('A request is already pending')
            if self.brain is None:raise ValueError('Qwen conversation worker unavailable; no request sent')
            self.task_service.submit(text,execute=execute)
            return self.state()
    def chat(self,text):
        return self.submit_task(text,execute=False)
    def preview(self,text):
        if not isinstance(text,str) or not text.strip() or len(text)>2000:raise ValueError('Request must be 1..2000 characters')
        with self.lock:
            if self._pending():raise ValueError('A request is already pending')
            if self.brain is None:raise ValueError('Brain unavailable; no request sent')
            brain_status=self.brain.status()
            if not brain_status.get('available',brain_status.get('configured',False)):raise ValueError('Brain worker unavailable; no request sent')
            if not hasattr(self.source,'ensure_brain_scene'):self.source.snapshot()
            self.generation+=1;epoch=self.generation;self.error=None
            self.task={'status':'planning','available':False,'reason':'preview_only','plan':None}
            self.worker=threading.Thread(target=self._preview,args=(epoch,text),daemon=True);self.worker.start()
            return self.state()
    def _preview(self,epoch,text):
        try:
            ensure=getattr(self.source,'ensure_brain_scene',None)
            scene=ensure() if ensure else self.source.brain_scene()
            result=self.brain.interpret(text,scene)
            self.source.snapshot()
            if not result.accepted or result.plan is None:raise ValueError('Plan rejected')
            plan=result.plan.to_dict()
            with self.lock:
                if epoch!=self.generation:return
                self.task.update(status='preview',plan=plan,provenance=getattr(result,'provenance','unknown'))
        except Exception:
            with self.lock:
                if epoch==self.generation:self.error='plan_rejected_or_scene_stale';self.task['status']='failed'
    def waypoint(self,object_id):
        with self.lock:
            if not self.enable_waypoint:raise ValueError('Waypoint experiment disabled pending runtime validation')
            if self._pending():raise ValueError('A request is already pending')
            vision_status=self.vision.status() if self.vision else {}
            if not vision_status.get('available') or not self._controller_status().get('available'):
                raise ValueError('Vision and SO101 Jev workers must both be available; no request sent')
            if object_id not in ('red','blue'):raise ValueError('Unsupported visual target')
            factory=self.executor_factory
            if factory is None:
                from embodied_arm.control.so101_executor import VisualWaypointExecutor
                factory=VisualWaypointExecutor
            self.waypoint_executor=factory(self.sim,self.source,self.controller)
            self.generation+=1;epoch=self.generation;self.error=None
            self.task.update(status='waypoint_experiment',plan=None)
            self.worker=threading.Thread(target=self._waypoint,args=(epoch,object_id),daemon=True);self.worker.start()
            return self.state()
    def _waypoint(self,epoch,object_id):
        try:result=self.waypoint_executor.run(object_id,[0,0,.035],max_calls=12,max_seconds=90)
        except Exception:result={'status':'stopped','reason':'waypoint_execution_failed'}
        with self.lock:
            if epoch!=self.generation:return
            self.task['status']='waypoint_'+result['status']
            if result['status']!='completed':self.error=result.get('reason','waypoint_stopped')
    def camera_pair(self):
        return self.camera.get()
    def control(self,request):
        if request.action=='reset' and not 0<=request.seed<=9999:raise ValueError('Seed must be 0..9999')
        with self.lock:
            self.generation+=1
            self.camera.invalidate()
            self.source.cancel()
            if self.waypoint_executor:self.waypoint_executor.stop()
            if self.brain and hasattr(self.brain,'cancel'):self.brain.cancel()
            self.task.update(status='stopped',plan=None);self.error=None
            if request.action=='reset':self.task_service.reset()
            else:self.task_service.stop()
            self.sim.stop()
            if request.action=='reset':
                self.sim.reset(request.seed);self.sim.stop();self.frames=None
            return self.state()
    def close(self):
        self.camera.close()
        with self.lock:
            self.closed=True;self.generation+=1;self.source.cancel();self.task_service.stop()
            if self.waypoint_executor:self.waypoint_executor.stop()
            if self.brain and hasattr(self.brain,'cancel'):self.brain.cancel()
            self.sim.stop();self.sim.close()

def create_app(sim_factory=None,vision=None,brain=None,source_factory=None,controller=None,enable_waypoint=False,executor_factory=None,task_executor=None,task_executor_factory=None,camera_interval_seconds=1.):
    @asynccontextmanager
    async def lifespan(app):
        factory=sim_factory
        if factory is None:
            from embodied_arm.simulation.so101_simulation import SO101Simulation
            factory=SO101Simulation
        vision_client=vision;brain_client=brain;controller_client=controller
        if sim_factory is None:
            if controller_client is None:
                from embodied_arm.runtime.so101_ipc import FileSO101Bridge
                controller_client=FileSO101Bridge()
            if vision_client is None:
                from embodied_arm.runtime.vision_ipc import FileVisionBridge
                vision_client=FileVisionBridge()
            if brain_client is None:
                from embodied_arm.runtime.brain_ipc import FileBrainBridge
                brain_client=FileBrainBridge()
        session=SO101Session(factory(seed=0),vision=vision_client,brain=brain_client,source_factory=source_factory,controller=controller_client,enable_waypoint=enable_waypoint,executor_factory=executor_factory,task_executor=task_executor,task_executor_factory=task_executor_factory,camera_interval_seconds=camera_interval_seconds);session.sim.stop();app.state.session=session
        session.camera.start()
        def physics():
            last=time.monotonic();accumulator=0.
            while not session.closed:
                now=time.monotonic();accumulator=min(.1,accumulator+now-last);last=now
                steps=min(24,int(accumulator*240))
                if steps:session.sim.tick(steps);accumulator-=steps/240
                time.sleep(.004)
        ticker=None
        if hasattr(session.sim,'tick'):
            ticker=threading.Thread(target=physics,daemon=True);ticker.start()
        yield
        session.closed=True
        if ticker:ticker.join(1)
        session.close()
    app=FastAPI(title='SO-ARM101 Image Lab',lifespan=lifespan)
    app.mount('/static',StaticFiles(directory=ROOT/'static_so101'),name='static')
    @app.get('/')
    def index():return FileResponse(ROOT/'static_so101'/'index.html')
    @app.get('/api/state')
    def state():return app.state.session.state()
    @app.get('/api/frames')
    def frames():
        try:return app.state.session.camera_pair()
        except ValueError as exc:raise HTTPException(503,detail=str(exc)) from None
    @app.post('/api/waypoint')
    def waypoint(request:WaypointRequest):
        try:return app.state.session.waypoint(request.object_id)
        except ValueError as exc:raise HTTPException(409,detail=str(exc)) from None
    @app.post('/api/observe')
    def observe():
        try:return app.state.session.observe()
        except ValueError as exc:raise HTTPException(409,detail=str(exc)) from None
    @app.post('/api/tasks')
    def tasks(request:TaskRequest):
        try:return app.state.session.submit_task(request.text,request.execute)
        except ValueError as exc:raise HTTPException(409,detail=str(exc)) from None
    @app.post('/api/chat')
    def chat(request:PlanRequest):
        try:return app.state.session.chat(request.text)
        except ValueError as exc:raise HTTPException(409,detail=str(exc)) from None
    @app.post('/api/plan')
    def plan(request:PlanRequest):
        try:return app.state.session.preview(request.text)
        except ValueError as exc:raise HTTPException(409,detail=str(exc)) from None
    @app.post('/api/control')
    def control(request:ControlRequest):
        try:return app.state.session.control(request)
        except ValueError as exc:raise HTTPException(409,detail=str(exc)) from None
    return app

app=create_app(enable_waypoint=os.environ.get('SO101_ENABLE_WAYPOINT')=='1',
    camera_interval_seconds=float(os.environ.get('SO101_CAMERA_INTERVAL','1')))
