"""Generic full-image dialogue/action UI; no legacy scene/CV/manipulation pipeline."""
from contextlib import asynccontextmanager
from pathlib import Path
import os
import threading
import time
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from embodied_arm.services.camera_sampler import CameraSampler
from embodied_arm.providers.visual_brain import sanitize_robot

ROOT=Path(__file__).resolve().parent
class TaskRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text:str
    execute:bool=True
class ControlRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action:Literal['stop','reset']

class UnconfiguredProvider:
    def __init__(self,component):self.component=component
    def status(self):return {'available':False,'configured':False,'calls':0,'component':self.component,'provenance':'unconfigured'}
    def cancel(self):pass

def create_app(sim_factory=None,qwen=None,jev=None,service_factory=None,*,live=False,camera_interval_seconds=1.,cycle_max_calls=32,mode="qwen_direct"):
    if mode not in ("qwen_direct","two_model"):raise ValueError("invalid_control_mode")
    @asynccontextmanager
    async def lifespan(app):
        factory=sim_factory
        if factory is None:
            from embodied_arm.simulation.so101_simulation import SO101Simulation
            factory=lambda:SO101Simulation(camera_profile='observer_wrist')
        sim=factory();sim.stop()
        qwen_client=qwen;jev_client=jev if mode=="two_model" else None
        if live:
            from embodied_arm.runtime.visual_loop_ipc import FileVisualBridge
            if qwen_client is None:qwen_client=FileVisualBridge(component='qwen')
            if mode=='two_model' and jev_client is None:jev_client=FileVisualBridge(component='jev')
        qwen_client=qwen_client or UnconfiguredProvider('qwen')
        if mode=='two_model':jev_client=jev_client or UnconfiguredProvider('jev')
        selected_service=service_factory
        if selected_service is None:
            from embodied_arm.services.visual_action_loop import VisualActionLoop
            selected_service=VisualActionLoop
        service=selected_service(sim,qwen_client,jev_client,cycle_max_calls=cycle_max_calls,max_calls=200,max_seconds=420,mode=mode)
        camera=CameraSampler(sim,interval_seconds=camera_interval_seconds)
        app.state.sim=sim;app.state.service=service;app.state.camera=camera
        app.state.qwen=qwen_client;app.state.jev=jev_client
        closing=threading.Event();camera.start()
        def physics():
            last=time.monotonic();accumulator=0.
            while not closing.is_set():
                now=time.monotonic();accumulator=min(.1,accumulator+now-last);last=now
                steps=min(24,int(accumulator*240))
                if steps:sim.tick(steps);accumulator-=steps/240
                closing.wait(.004)
        ticker=None
        if hasattr(sim,'tick'):
            ticker=threading.Thread(target=physics,daemon=True);ticker.start()
        try:yield
        finally:
            settled=service.close();closing.set()
            if ticker:ticker.join(2)
            camera.close()
            if settled is not False:sim.close()
            else:
                # A late provider must settle before its simulator is destroyed.
                def deferred_cleanup():
                    if service.wait_closed(65):sim.close()
                threading.Thread(target=deferred_cleanup,daemon=True,name='visual-deferred-cleanup').start()
    app=FastAPI(title='SO-ARM101 Visual Action',lifespan=lifespan)
    app.mount('/static',StaticFiles(directory=ROOT/'static_visual'),name='static')
    def state():
        return {'schema':'visual-action-ui-v1','robot':sanitize_robot(app.state.sim.proprioception()),
                'camera':app.state.camera.status(),'workflow':app.state.service.status(),
                'qwen':app.state.qwen.status(),'jev':app.state.jev.status() if app.state.jev is not None else {'available':False,'enabled':False,'calls':0}}
    @app.get('/')
    def index():return FileResponse(ROOT/'static_visual'/'index.html')
    app.get('/api/state')(state)
    @app.get('/api/frames')
    def frames():
        try:return app.state.camera.get()
        except ValueError:raise HTTPException(503,detail='camera_unavailable') from None
    @app.post('/api/tasks')
    def submit(request:TaskRequest):
        try:app.state.service.submit(request.text,execute=request.execute)
        except ValueError as exc:
            reason=str(exc)
            if reason not in {'task_busy','invalid_request','invalid_text','invalid_mode','qwen_unavailable','brain_unavailable','jev_unavailable','execution_unavailable','provider_unavailable'}:reason='request_rejected'
            raise HTTPException(409,detail=reason) from None
        return state()
    @app.post('/api/control')
    def control(request:ControlRequest):
        if request.action=='stop':app.state.service.stop()
        else:
            if app.state.service.status().get('pending'):raise HTTPException(409,detail='task_busy')
            app.state.service.reset();app.state.camera.invalidate()
        return state()
    return app

app=create_app(live=os.environ.get('SO101_VISUAL_LIVE')=='1',camera_interval_seconds=float(os.environ.get('SO101_CAMERA_INTERVAL','1')),cycle_max_calls=int(os.environ.get('SO101_CYCLE_MAX_CALLS','32')),mode=os.environ.get('SO101_CONTROL_MODE','qwen_direct'))
