"""Real browser + real HTTP/app, explicitly fake providers. Never live API evidence."""
import io
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from PIL import Image

from embodied_arm.web.app_so101 import create_app
from embodied_arm.providers.brain import BrainPlan, BrainResult


class FakeRobot:
    """No physics or object truth. This suite validates application behavior only."""
    def __init__(self, seed=0): self.epoch = 0; self.paused = True; self.captures = 0
    def proprioception(self):
        return dict(name='FAKE E2E SO101', joints=[0]*5, ee=[0,0,0],
                    gripper_openness=1, paused=self.paused, busy=False, actions=0)
    def capture_frames(self):
        import base64
        self.captures += 1
        buf = io.BytesIO(); Image.new('RGB', (32,32), 'gray').save(buf, 'JPEG')
        image = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
        return dict(frame_id=f'FAKE-E2E-{self.captures}', captured_at=time.monotonic(), images=[
            dict(camera_id=c, data_url=image) for c in ('left','right')])
    def evaluation_state(self): raise AssertionError('Oracle access forbidden')
    def stop(self): self.paused = True
    def reset(self, seed): self.epoch += 1
    def close(self): pass


class FakeBrain:
    def __init__(self):
        self.available=True; self.calls=[]; self.gate=threading.Event(); self.gate.set()
        self.fail=False
    def status(self):
        return dict(available=self.available, configured=self.available,
                    model='FAKE-E2E-NOT-QWEN', calls=len(self.calls))
    def interpret(self, text, scene, history=None):
        self.calls.append((text,scene,history)); self.gate.wait(30)
        if self.fail: raise RuntimeError('Deliberate fake provider failure')
        return BrainResult(accepted=True, provenance='FAKE-E2E-NOT-LIVE',
            plan=BrainPlan(action='pick',object_id='red',message='FAKE：测试任务，非真实抓取。') if scene.get('fixture_task') else BrainPlan(action='clarify', message='FAKE：当前没有视觉观察，不能确认物体或执行抓取。'))
    def cancel(self): pass  # Deliberately allows late response to test generation guard.


class FakeTaskSource:
    """Synthetic scene contract only: not pixel/physics/vision validation."""
    def __init__(self, sim, vision=None): self.scene=None
    def refresh(self): self.scene={'captured_at':time.monotonic(),'objects':{}}
    def status(self): return {'scene':self.scene,'reason':'FAKE-E2E','max_position_std_m':.003}
    def snapshot(self):
        if self.scene is None: raise ValueError('No fake observation')
        return {'fixture_task':True}
    def brain_scene(self): return self.snapshot()
    def cancel(self): self.scene=None


class FakeTaskExecutor:
    """Fake acceptance flag is ONLY for application lifecycle tests."""
    validated=True
    def __init__(self):
        self.gate=threading.Event();self.gate.set();self.verified=True;self.calls=0;self.stops=0
    def execute(self,plan,*,cancel_event,event_callback):
        self.calls+=1;event_callback('FAKE-E2E progress; no robot motion')
        self.gate.wait(30)
        return {'verified':self.verified,'provenance':'FAKE-E2E-NOT-PHYSICAL'}
    def stop(self): self.stops+=1


@pytest.fixture(scope='module')
def browser():
    if os.environ.get('E2E_BROWSER') != '1':
        pytest.skip('Opt-in browser suite: set E2E_BROWSER=1; skips are not passes')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        executable = os.environ.get('E2E_CHROMIUM_PATH')
        b=p.chromium.launch(headless=True, **({'executable_path':executable} if executable else {}))
        yield b
        b.close()


@pytest.fixture
def lab(browser, request):
    brain=FakeBrain(); executor=FakeTaskExecutor() if getattr(request,'param',False) else None
    app=create_app(FakeRobot, brain=brain, **({'source_factory':FakeTaskSource,'task_executor':executor} if executor else {}))
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='error'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
    deadline=time.monotonic()+10
    while not server.started:
        if time.monotonic()>deadline: raise RuntimeError('Local test server did not start')
        time.sleep(.02)
    context=browser.new_context(viewport={'width':1440,'height':1000})
    page=context.new_page(); errors=[]; page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(f'http://127.0.0.1:{port}')
    page.wait_for_function("!document.getElementById('send').disabled")
    yield SimpleNamespace(page=page,brain=brain,app=app,errors=errors,executor=executor)
    brain.gate.set();
    if executor: executor.gate.set()
    context.close(); server.should_exit=True;thread.join(5);sock.close()
    assert not errors, f'Browser JavaScript errors: {errors}'


def wait_status(lab,status):
    lab.page.wait_for_function("async status => (await (await fetch('/api/state')).json()).chat.status === status", arg=status)
    lab.page.wait_for_timeout(1100) # Let the actual UI polling render, rather than invoking render directly.


def send(lab,text='你好'):
    lab.page.locator('#request').fill(text);lab.page.locator('#send').click()


def test_text_reply_and_unknown_image_are_honest(lab):
    from playwright.sync_api import expect
    send(lab,'你好')
    wait_status(lab,'ready')
    expect(lab.page.locator('#chat-history')).to_contain_text('没有视觉观察')
    expect(lab.page.locator('#chat-status')).to_contain_text('FAKE-E2E-NOT-LIVE')
    assert lab.brain.calls[0][1] == {'observation_status':'unavailable','blocks':{}}
    expect(lab.page.locator('#plan-send')).to_be_disabled()
    assert lab.app.state.session.sim.proprioception()['actions']==0


def test_enter_shift_enter_and_double_click(lab):
    lab.brain.gate.clear()
    field=lab.page.locator('#request');field.fill('你好');field.press('Shift+Enter')
    assert len(lab.brain.calls)==0
    field.press('Enter')
    lab.page.wait_for_function("document.getElementById('send').disabled")
    # DOM-level rapid repeats exercise the handler's in-flight guard without waiting for enablement.
    lab.page.locator('#send').evaluate('(button) => { button.click(); button.click(); }')
    assert len(lab.brain.calls)==1
    lab.brain.gate.set();wait_status(lab,'ready')


def test_stop_pending_discards_late_reply(lab):
    from playwright.sync_api import expect
    lab.brain.gate.clear();send(lab);wait_status(lab,'thinking')
    lab.page.locator('#stop').click();wait_status(lab,'cancelled')
    lab.brain.gate.set();lab.page.wait_for_timeout(1300)
    expect(lab.page.locator('#chat-history .assistant')).to_have_count(0)
    expect(lab.page.locator('#robot-status')).to_have_text('已停止')
    expect(lab.page.locator('#send')).to_be_enabled()


def test_provider_error_is_truthful_and_retry_works(lab):
    from playwright.sync_api import expect
    lab.brain.fail=True;send(lab);wait_status(lab,'error')
    expect(lab.page.locator('#chat-status')).to_contain_text('没有执行动作')
    expect(lab.page.locator('#chat-history .assistant')).to_have_count(0)
    lab.brain.fail=False;send(lab,'你好');wait_status(lab,'ready')
    assert len(lab.brain.calls)==2
    expect(lab.page.locator('#chat-history .assistant')).to_have_count(1)


def test_refresh_during_request_recovers_history_without_resubmit(lab):
    from playwright.sync_api import expect
    lab.brain.gate.clear();send(lab,'你好');wait_status(lab,'thinking')
    lab.page.reload();expect(lab.page.locator('#send')).to_be_disabled()
    lab.brain.gate.set();wait_status(lab,'ready')
    expect(lab.page.locator('#chat-history')).to_contain_text('你好')
    assert len(lab.brain.calls)==1


def test_expired_service_and_reconnect_preserve_draft(lab):
    from playwright.sync_api import expect
    lab.page.locator('#request').fill('恢复连接后发送')
    lab.brain.available=False
    expect(lab.page.locator('#send')).to_be_disabled(timeout=4000)
    expect(lab.page.locator('#chat-status')).to_contain_text('离线')
    expect(lab.page.locator('#request')).to_be_enabled()
    lab.brain.available=True
    expect(lab.page.locator('#send')).to_be_enabled(timeout=4000)
    expect(lab.page.locator('#request')).to_have_value('恢复连接后发送')
    lab.page.locator('#request').fill('你好')
    lab.page.locator('#send').click();wait_status(lab,'ready')


def test_task_request_does_not_claim_unimplemented_execution(lab):
    """Current safety contract, NOT acceptance of the requested autonomous task feature."""
    from playwright.sync_api import expect
    send(lab,'夹起红色方块');wait_status(lab,'error')
    expect(lab.page.locator('#execution-readiness')).to_contain_text('自动执行尚未验证')
    expect(lab.page.locator('#task-send')).to_be_disabled()
    expect(lab.page.locator('#plan-send')).to_be_disabled()
    assert lab.app.state.session.sim.proprioception()['actions']==0


@pytest.mark.parametrize('lab',[True],indirect=True)
def test_fake_task_progress_and_verified_completion(lab):
    from playwright.sync_api import expect
    lab.executor.gate.clear()
    lab.page.locator('#request').fill('夹起红色方块')
    lab.page.locator('#task-send').click()
    expect(lab.page.locator('#workflow-status')).to_contain_text('正在执行')
    expect(lab.page.locator('#workflow-events')).to_contain_text('FAKE-E2E progress')
    assert lab.executor.calls==1
    lab.executor.gate.set()
    expect(lab.page.locator('#workflow-status')).to_contain_text('任务流程结束')
    assert lab.app.state.session.task_service.status()['outcome']['provenance']=='FAKE-E2E-NOT-PHYSICAL'


@pytest.mark.parametrize('lab',[True],indirect=True)
def test_fake_task_unverified_outcome_is_failure(lab):
    from playwright.sync_api import expect
    lab.executor.verified=False
    lab.page.locator('#request').fill('夹起红色方块');lab.page.locator('#task-send').click()
    expect(lab.page.locator('#workflow-status')).to_contain_text('任务失败')
    assert lab.app.state.session.task_service.status()['state']=='failed'
    assert lab.executor.stops>=1


@pytest.mark.parametrize('lab',[True],indirect=True)
def test_stop_during_fake_execution_discards_late_success(lab):
    from playwright.sync_api import expect
    lab.executor.gate.clear()
    lab.page.locator('#request').fill('夹起红色方块');lab.page.locator('#task-send').click()
    expect(lab.page.locator('#workflow-status')).to_contain_text('正在执行')
    lab.page.locator('#stop').click()
    lab.executor.gate.set()
    expect(lab.page.locator('#workflow-status')).to_contain_text('任务已停止')
    assert lab.app.state.session.task_service.status()['state']=='cancelled'
    assert lab.executor.stops>=1
