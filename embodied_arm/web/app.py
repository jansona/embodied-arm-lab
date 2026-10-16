import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
import threading
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from embodied_arm.simulation.simulator import Simulation
from embodied_arm.providers.adapters import request_preview, ALIYUN_ENDPOINT, ALIYUN_MODEL
from embodied_arm.legacy.panda.control import observation
from embodied_arm.legacy.panda.runner import JevRunner
from embodied_arm.runtime.ipc import FileBridgeAdapter
from embodied_arm.legacy.panda.task_executor import TaskExecutor, plan_from_intent
from embodied_arm.legacy.panda.sequence_executor import SequenceExecutor

from embodied_arm.paths import WORKSPACE_ROOT, WEB_ROOT
ROOT=WORKSPACE_ROOT
sim=None
runner=None
brain=None
executor=None
chat_lock=threading.RLock()
chat_epoch=0
chat_thread=None
chat_data={"status":"idle","messages":[],"plan":None,"error":None,"controller":"deterministic","provenance":"none"}
CONFIG_PATH=ROOT.parent / 'jev-arm-config.json'
RECOVERY_PATH=ROOT/'artifacts'/'validated-plan-recovery.json'
IPC_MODE=os.environ.get('JEV_USE_IPC')=='1'
LIVE_PROVIDER_READY=True  # verified Aliyun System One endpoint/model; explicit config and Start required
halt=threading.Event()

def loop():
    previous=time.monotonic();accumulator=0.0
    while not halt.is_set():
        start=time.monotonic()
        # Fixed 1/240 physics steps, bounded wall-clock catch-up after rendering.
        accumulator=min(.1,accumulator+max(0,start-previous));previous=start
        steps=min(24,int(accumulator*240))
        if steps:
            sim.tick(steps);accumulator-=steps/240
        halt.wait(max(0,1/60-(time.monotonic()-start)))

@asynccontextmanager
async def lifespan(app):
    global sim,runner,brain,executor,chat_thread,chat_epoch,chat_data
    sim=Simulation();runner=JevRunner(sim,adapter=FileBridgeAdapter() if IPC_MODE else None,use_environment_network=os.environ.get('JEV_USE_ENVIRONMENT_NETWORK')=='1');halt.clear()
    brain=None;executor=TaskExecutor(sim);chat_thread=None;chat_epoch+=1
    chat_data={"status":"idle","messages":[],"plan":None,"error":None,"controller":"deterministic","provenance":"none"}
    worker=threading.Thread(target=loop,daemon=True);worker.start()
    yield
    cancel_chat();runner.stop();halt.set();worker.join(timeout=3);sim.close()

app=FastAPI(lifespan=lifespan,docs_url='/docs')

@app.middleware('http')
async def local_only(request:Request,call_next):
    # Bound to loopback, no remote URL or credential entry in this UI.
    # Browser cross-origin requests may read nothing and may not mutate state.
    origin=request.headers.get('origin')
    if request.method!='GET' and origin and origin not in {'http://127.0.0.1:8765','http://localhost:8765','http://127.0.0.1:8766','http://localhost:8766'}:
        return Response('Cross-origin control denied',status_code=403)
    response=await call_next(request)
    response.headers['Cache-Control']='no-store'
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Content-Security-Policy']="default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'"
    return response

@app.get('/')
def home():return FileResponse(WEB_ROOT/'static/index.html')
@app.get('/ui.js')
def js():return FileResponse(WEB_ROOT/'static/ui.js',media_type='text/javascript')
@app.get('/api/state')
def state():
    return {**sim.state(),'chat':chat_status(),'recovery':recovery_status(),'jev':{**runner.status(),'supports_images':False,**({'worker':runner._adapter.status()} if IPC_MODE else {})}}
@app.get('/api/frame')
def frame():return Response(sim.render(),media_type='image/jpeg')
@app.get('/api/jev/preview')
def preview():return request_preview(observation(sim),endpoint=ALIYUN_ENDPOINT,model=ALIYUN_MODEL)
class Command(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    action:str
    values:list[float]|None=None
    seed:int=0
@app.post('/api/control')
def control(command:Command):
    try:
        if command.action in ('reset','pause','stop'):
            cancel_chat()
        elif chat_status()['busy']:
            raise ValueError('Stop the dialogue task before manual control')
        if command.action=='reset':
            runner.stop();sim.reset(command.seed)
        elif command.action=='baseline':
            if runner.status().get('running'):raise ValueError('Stop Jev before using the scripted baseline')
            sim.baseline()
        elif command.action=='pause':
            if runner.status().get('running'):runner.pause()
            with sim.lock:sim.paused=True;sim.status='Paused'
        elif command.action=='resume':
            with sim.lock:sim.paused=False;sim.status='Running' if sim.current else 'Ready'
        elif command.action=='stop':runner.stop();sim.stop()
        elif command.action=='jev':raise ValueError('Jev is not configured. No request was sent.')
        else:
            if runner.status().get('running'):raise ValueError('Stop Jev before manual control')
            sim.manual(command.action,command.values)
    except (ValueError,TypeError) as e:raise HTTPException(409,detail=str(e))
    return state()

class JevRun(BaseModel):
    model_config=ConfigDict(extra='forbid')
    max_calls:int=10
    max_seconds:float=30

@app.post('/api/jev/configure')
def configure_jev():
    if chat_status()['busy']:raise HTTPException(409,detail='Stop the dialogue task first')
    if IPC_MODE:return state()
    if not LIVE_PROVIDER_READY:raise HTTPException(409,detail='Aliyun provider verification pending; no configuration loaded and no API request sent')
    try:runner.configure(CONFIG_PATH)
    except (ValueError,OSError) as e:raise HTTPException(409,detail=str(e))
    return state()

@app.post('/api/jev/start')
def start_jev(request:JevRun):
    if chat_status()['busy']:raise HTTPException(409,detail='Stop the dialogue task first')
    if not LIVE_PROVIDER_READY:raise HTTPException(409,detail='Aliyun provider verification pending; no API request sent')
    try:runner.start(max_calls=request.max_calls,max_seconds=request.max_seconds)
    except (ValueError,RuntimeError) as e:raise HTTPException(409,detail=str(e))
    return state()

@app.get('/api/jev/connectivity')
def check_connectivity():
    from embodied_arm.providers.connectivity import check
    return check()

@app.post('/api/jev/smoke')
def smoke_test():
    if chat_status()['busy']:raise HTTPException(409,detail='Stop the dialogue task first')
    if IPC_MODE:raise HTTPException(409,detail='Use the API-only CLI in the managed environment; this UI worker accepts robot observations only')
    # API-only validation. No robot observation, planning or actuation occurs.
    runner.stop()
    if not runner.status()['configured']:
        raise HTTPException(409,detail='Load the approved Aliyun config first')
    with runner._lock:
        if runner._thread and runner._thread.is_alive():
            raise HTTPException(409,detail='Wait for the pending request to finish')
        adapter=runner._adapter
    return adapter.smoke_test()


# One outstanding dialogue task; epochs make Stop/Reset invalidate late replies.
def chat_status():
    with chat_lock:
        task=executor.status() if executor else {'status':'idle'}
        status=chat_data['status']
        if status=='executing' and task['status']!='running':status=task['status']
        outcome=task.get('outcome')
        plan=chat_data.get('plan')
        if status=='completed' and plan and plan.get('object_id'):
            scene=sim.state();position=scene['blocks'][plan['object_id']]
            if plan.get('target_id')=='goal':
                import math
                success=math.dist(position[:2],scene['goal'][:2])<.055 and abs(position[2]-.025)<.025
            else:success=position[2]>.13
            outcome={'verified':success,'description':'目标状态已达到' if success else '动作结束，但未达到目标状态'}
        brain_status=brain.status() if brain else {'configured':False,'model':'qwen3.8-max','calls':0}
        brain_status={**brain_status,'calls':max(brain_status.get('calls',0),chat_data.get('brain_calls',0),task.get('brain_calls',0))}
        return {**chat_data,'outcome':outcome,'messages':[dict(m) for m in chat_data['messages']],
                'status':status,'task':task,
                'busy':status=='thinking' or task['status'] in ('running','replanning'),
                'request_pending':bool(chat_thread and chat_thread.is_alive()),
                'brain':brain_status}


def cancel_chat():
    global chat_epoch
    with chat_lock:
        chat_epoch+=1
        if executor:executor.stop()
        if brain and hasattr(brain,'cancel'):brain.cancel()
        chat_data.update(status='stopped',error=None)


class BrainConfiguration(BaseModel):
    model_config=ConfigDict(extra='forbid')
    mode:Literal['live','mock']='live'


@app.post('/api/brain/configure')
def configure_brain(request:BrainConfiguration):
    global brain
    with chat_lock:
        if chat_status()['busy'] or (chat_thread and chat_thread.is_alive()):
            raise HTTPException(409,detail='Stop and wait for the pending dialogue request')
        try:
            from embodied_arm.providers.brain import QwenBrain, MockBrain
            if request.mode=='mock':brain=MockBrain()
            else:
                if IPC_MODE:
                    from embodied_arm.runtime.brain_ipc import FileBrainBridge
                    brain=FileBrainBridge()
                else:
                    runner.configure(CONFIG_PATH)
                    if not runner.status()['configured']:raise ValueError('Server configuration is unavailable')
                    brain=QwenBrain(api_key=runner._adapter._api_key,use_environment_network=os.environ.get('JEV_USE_ENVIRONMENT_NETWORK')=='1')
            chat_data.update(status='ready',error=None,plan=None,provenance=request.mode,brain_calls=0,error_code=None)
        except Exception:
            raise HTTPException(409,detail='Brain configuration unavailable; no request sent. Check the server configuration and runtime mode.') from None
    return state()


class ChatRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    text:str=Field(min_length=1,max_length=2000)
    controller:Literal['deterministic','jev']='deterministic'
    allow_replan:bool=False


@app.post('/api/chat')
def chat(request:ChatRequest):
    global chat_epoch,chat_thread,executor
    with chat_lock:
        if brain is None:raise HTTPException(409,detail='Choose and configure the brain first')
        if chat_status()['busy'] or (chat_thread and chat_thread.is_alive()) or runner.status()['pending'] or sim.state()['busy']:
            raise HTTPException(409,detail='A task or request is pending; stop it and wait before sending again')
        if request.controller=='jev' and not runner.status()['configured']:
            raise HTTPException(409,detail='Jev controller is not configured; select deterministic explicitly or configure Jev')
        if request.controller=='jev' and IPC_MODE and not runner._adapter.status().get('available'):
            raise HTTPException(409,detail='Jev worker is unavailable; no brain request was sent')
        chat_epoch+=1;epoch=chat_epoch
        executor=TaskExecutor(sim,controller=runner._adapter if request.controller=='jev' else None)
        chat_data['messages'].append({'role':'user','content':request.text})
        chat_data['messages']=chat_data['messages'][-40:]
        chat_data.update(status='thinking',error=None,error_code=None,plan=None,controller=request.controller)
        worker=threading.Thread(target=interpret_chat,args=(epoch,request.text,request.controller,request.allow_replan),daemon=True)
        chat_thread=worker;worker.start()
    return state()


def interpret_chat(epoch,text,controller,allow_replan=False):
    global executor
    try:
        with chat_lock:
            active_brain=brain;history=[dict(m) for m in chat_data['messages'][:-1]]
        result=active_brain.interpret(text,sim.state(),history=history)
        with chat_lock:
            if epoch!=chat_epoch:return
            chat_data['brain_calls']=max(chat_data.get('brain_calls',0),result.calls)
            if not result.accepted or result.plan is None:
                reason=result.reason
                from embodied_arm.providers.brain import PROVIDER_ERROR_REASONS, REPLAN_ERROR_REASONS
                known={'busy','invalid_input','cancelled','unavailable','expired','timeout','invalid_response','call_budget_exhausted','response_too_large','unexpected_model','incomplete_response','network_error'} | PROVIDER_ERROR_REASONS | REPLAN_ERROR_REASONS
                if reason not in known and not (reason.startswith('http_error_') and reason[11:].isdigit() and len(reason)==14):reason='unavailable'
                chat_data.update(status='error',error_code=reason,error='大脑未返回有效指令，机械臂未执行。错误：'+reason)
                return
            plan=result.plan.to_dict()
            chat_data.update(plan=plan,provenance=getattr(result,'provenance',chat_data['provenance']))
            chat_data['messages'].append({'role':'assistant','content':plan['message']})
            if plan['action']=='stop':
                executor.stop();runner.stop();chat_data['status']='stopped'
            elif plan['action'] in ('status','clarify'):
                chat_data['status']=plan['action']
            elif plan['action']=='sequence':
                executor=SequenceExecutor(sim,controller=runner._adapter if controller=='jev' else None,
                    brain=active_brain,text=text,history=history,allow_replan=allow_replan)
                executor.start(plan);chat_data['status']='executing'
            else:
                grounded=plan_from_intent(plan,sim.state())
                executor=TaskExecutor(sim,controller=runner._adapter if controller=='jev' else None)
                executor.start(grounded)
                chat_data['status']='executing'
    except Exception:
        with chat_lock:
            if epoch==chat_epoch:chat_data.update(status='error',error='任务未能执行；请检查目标、控制器和当前状态。没有切换到其他模型或控制器。')


@app.post('/api/chat/stop')
def stop_chat():
    cancel_chat();runner.stop();sim.stop()
    return state()



def read_recovery_plan():
    from embodied_arm.providers.brain import validate_plan
    try:
        raw=RECOVERY_PATH.read_bytes()
        if len(raw)>16384:raise ValueError('Oversized recovery artifact')
        record=json.loads(raw)
        if (not isinstance(record,dict) or set(record)!={'plan','source_call_id','model','provenance','accepted'}
            or record['model']!='qwen3.8-max' or record['provenance']!='live' or record['accepted'] is not True
            or not isinstance(record['source_call_id'],str) or not 1<=len(record['source_call_id'])<=200):
            raise ValueError('Invalid recovery provenance')
        plan=validate_plan(record['plan'],sim.state(),check_state=False).to_dict()
        approved_steps=[{'step_id':'s1','action':'pick','object_id':'red','target_id':None},
                        {'step_id':'s2','action':'place','object_id':'red','target_id':'goal'},
                        {'step_id':'s3','action':'pick','object_id':'blue','target_id':None},
                        {'step_id':'s4','action':'place','object_id':'blue','target_id':'on:red'}]
        single=plan['action']=='pick' and plan['object_id']=='red' and plan['target_id'] is None
        sequence=plan['action']=='sequence' and plan.get('steps')==approved_steps
        if not (single or sequence):raise ValueError('Recovery is limited to approved exact plans')
        return {**record,'plan':plan}
    except (OSError,ValueError,TypeError,KeyError,RecursionError):
        raise ValueError('No approved recovery plan is available') from None


def recovery_status():
    try:
        record=read_recovery_plan()
        return {'available':True,'model':record['model'],'new_brain_calls':0,'controller':'jev','action':record['plan']['action']}
    except ValueError:return {'available':False}


@app.post('/api/chat/recover')
def recover_chat():
    """One-use replay of the approved saved plan; never starts brain inference."""
    global executor,chat_epoch
    with chat_lock:
        if chat_status()['busy'] or (chat_thread and chat_thread.is_alive()) or runner.status()['pending']:
            raise HTTPException(409,detail='Stop and wait for the current request first')
        if not runner.status()['configured']:
            raise HTTPException(409,detail='Jev is not configured; recovery cannot fall back')
        if IPC_MODE and not runner._adapter.status().get('available'):
            raise HTTPException(409,detail='Jev worker is unavailable; no recovery started')
        try:
            record=read_recovery_plan()
            # Consume once before any actuation. A further replay requires a new
            # explicitly prepared artifact rather than repeated browser clicks.
            RECOVERY_PATH.replace(RECOVERY_PATH.with_suffix('.consumed.json'))
            chat_epoch+=1;runner.stop();sim.reset(0)
            if record['plan']['action']=='sequence':
                grounded=record['plan']
                executor=SequenceExecutor(sim,controller=runner._adapter,allow_replan=False,
                                          max_controller_calls=200,max_seconds=420)
            else:
                grounded=plan_from_intent(record['plan'],sim.state())
                executor=TaskExecutor(sim,controller=runner._adapter)
            chat_data.update(status='executing',error=None,error_code=None,plan=record['plan'],
                             controller='jev',provenance='reused_live',brain_calls=0)
            chat_data['messages'].append({'role':'assistant','content':'复用先前 qwen3.8-max 已验证的原始计划；未重新调用大脑。明确从新场景开始，由 Jev 执行全部原始步骤。'})
            executor.start(grounded)
        except (ValueError,OSError,RuntimeError):
            chat_data.update(status='error',error='恢复未能启动；没有调用大脑或切换控制器。')
            raise HTTPException(409,detail='Recovery could not start; no brain call or fallback') from None
    return state()
