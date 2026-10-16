import sys
import pytest
from start_so101 import ROOT, commands


def test_preview_starts_only_loopback_server():
    assert commands() == [[sys.executable, '-m', 'uvicorn', 'app_so101:app',
                           '--host', '127.0.0.1', '--port', '8766']]


def test_live_is_explicit_and_bounded(tmp_path):
    config = str(tmp_path / 'provider.json')
    jobs = commands(live=True, config=config)
    assert len(jobs) == 4
    assert jobs[1:] == commands(live=True, config=config, workers_only=True)
    assert [job[1] for job in jobs[1:]] == ['brain_worker.py', 'vision_worker.py', 'so101_worker.py']
    assert [job[-4:] for job in jobs[1:]] == [
        ['--max-calls', '6', '--max-seconds', '600'],
        ['--max-calls', '6', '--max-seconds', '600'],
        ['--max-calls', '12', '--max-seconds', '180']]
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
    import start_so101
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
    import start_so101
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
    import start_so101
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
