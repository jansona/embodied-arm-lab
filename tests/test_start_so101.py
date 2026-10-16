import sys
import pytest
from embodied_arm.runtime.start_so101 import ROOT, commands


def test_preview_starts_only_loopback_server():
    assert commands() == [[sys.executable, '-m', 'uvicorn', 'embodied_arm.web.app_so101:app',
                           '--host', '127.0.0.1', '--port', '8766']]


def test_live_is_explicit_and_bounded(tmp_path):
    config = str(tmp_path / 'provider.json')
    jobs = commands(live=True, config=config)
    assert len(jobs) == 4
    assert jobs[1:] == commands(live=True, config=config, workers_only=True)
    assert [job[1:3] for job in jobs[1:]] == [['-m', 'embodied_arm.runtime.' + name] for name in ['brain_worker', 'vision_worker', 'so101_worker']]
    assert [job[job.index('--max-calls'):job.index('--max-calls')+4] for job in jobs[1:]] == [
        ['--max-calls', '10', '--max-seconds', '600'],
        ['--max-calls', '10', '--max-seconds', '600'],
        ['--max-calls', '200', '--max-seconds', '420']]
    assert jobs[1][-2:] == ['--max-replans', '8']
    assert all('--max-replans' not in job for job in jobs[2:])
    assert not (tmp_path / 'provider.json').exists()  # Builder never opens credentials.


@pytest.mark.parametrize('kwargs', [
    {'live': True}, {'workers_only': True}, {'config': '/tmp/example.json'},
    {'live': True, 'config': str(ROOT / 'config.local.json')},
    {'port': 0}, {'port': 65536}, {'port': True},
])
def test_invalid_launch_is_rejected_before_process_start(kwargs):
    with pytest.raises(ValueError):
        commands(**kwargs)


def test_failed_worker_closes_started_processes(monkeypatch, tmp_path):
    import embodied_arm.runtime.start_so101 as start_so101
    children = []
    class Child:
        def __init__(self, job, **kwargs):
            self.index = len(children)
            self.terminated = False
            self.waited = False
            children.append(self)
        def poll(self):
            return 1 if self.index == 2 else (0 if self.terminated else None)
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            self.waited = True
            return self.poll()
    monkeypatch.setattr(start_so101.subprocess, 'Popen', Child)
    monkeypatch.setattr(start_so101.signal, 'signal', lambda *_: None)
    assert start_so101.main(['--live', '--config', str(tmp_path / 'external.json')]) == 1
    assert len(children) == 4
    assert all(child.waited for child in children)
    assert all(child.terminated for child in children if child.index != 2)


def test_partial_start_failure_closes_prior_process(monkeypatch):
    import embodied_arm.runtime.start_so101 as start_so101
    children = []
    class Child:
        def __init__(self, job, **kwargs):
            if children:
                raise OSError('simulated spawn failure')
            self.terminated = False
            children.append(self)
        def poll(self):
            return 0 if self.terminated else None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(start_so101.subprocess, 'Popen', Child)
    monkeypatch.setattr(start_so101.signal, 'signal', lambda *_: None)
    with pytest.raises(OSError):
        start_so101.main(['--live', '--config', '/tmp/external.json'])
    assert len(children) == 1 and children[0].terminated


def test_workers_only_expiry_does_not_restart(monkeypatch, tmp_path):
    import embodied_arm.runtime.start_so101 as start_so101
    children = []
    class Child:
        def __init__(self, *args, **kwargs):
            children.append(self)
        def poll(self):
            return 0
        def wait(self, timeout=None):
            return 0
    monkeypatch.setattr(start_so101.subprocess, 'Popen', Child)
    monkeypatch.setattr(start_so101.signal, 'signal', lambda *_: None)
    assert start_so101.main(['--live', '--workers-only', '--config', str(tmp_path / 'external.json')]) == 0
    assert len(children) == 3

@pytest.mark.parametrize('value',['0','0.1','11','nan'])
def test_invalid_camera_interval_fails_before_launch(monkeypatch,value):
 import embodied_arm.runtime.start_so101 as launcher
 monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('Must not launch'))
 with pytest.raises(SystemExit):launcher.main(['--camera-interval',value])


@pytest.mark.parametrize('live',[False,True])
def test_simulation_task_switch_and_notice_are_explicit(monkeypatch,tmp_path,capsys,live):
    import embodied_arm.runtime.start_so101 as launcher
    launched=[]
    class Child:
        def __init__(self,command,**kwargs):
            launched.append((command,kwargs))
        def poll(self):return 0
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(launcher.subprocess,'Popen',Child)
    monkeypatch.setattr(launcher.signal,'signal',lambda *_:None)
    # Neither an inherited enable nor an inherited disable overrides this launch.
    monkeypatch.setenv('SO101_SIMULATION_TASKS','0' if live else '1')
    args=['--live','--config',str(tmp_path/'external.json')] if live else []
    assert launcher.main(args)==0
    assert len(launched)==(4 if live else 1)
    assert all(kwargs['env']['SO101_SIMULATION_TASKS']==('1' if live else '0') for _,kwargs in launched)
    output=capsys.readouterr().out
    if live:
        assert '200 calls/420s' in output and '200 Jev calls/420s per task' in output
        assert '16 calls, at most 8 reviews' in output
        assert 'including idle time' in output and 'No automatic calls' in output
        assert 'runtime guards may stop' in output and 'No physical hardware control' in output
    else:
        assert 'task execution disabled' in output
    assert not (tmp_path/'external.json').exists()


def test_explicit_simulation_budget_fits_worker_hard_caps(tmp_path):
    from embodied_arm.runtime.so101_ipc import MAX_CALLS,MAX_SECONDS
    job=commands(live=True,config=str(tmp_path/'external.json'))[-1]
    assert int(job[job.index('--max-calls')+1])<=MAX_CALLS
    assert int(job[job.index('--max-seconds')+1])<=MAX_SECONDS
