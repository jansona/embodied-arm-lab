"""Offline worker/IPC replan budgets, using bounded mock workers only."""
import fcntl
import os
import threading
import time
from unittest.mock import patch

import pytest
from embodied_arm.providers.brain import BrainPlan, BrainResult, validate_cycle_context
from embodied_arm.runtime import brain_ipc, brain_worker

SCENE = {'blocks': {'red': [.2, 0, .01]}, 'ee': [.2, 0, .1], 'goal': [.3, 0, .01]}
PLAN = BrainPlan('pick', 'red', None, '尝试抓取')
FEEDBACK = {'completed_step_ids': [], 'failed_step_id': 's1', 'reason': 'cycle_budget'}


class OfflineBrain:
    def interpret(self, *args, **kwargs):
        return BrainResult(plan=PLAN, accepted=True, reason='accepted', calls=1, provenance='mock')

    replan = interpret


@pytest.fixture
def isolated_ipc(tmp_path, monkeypatch):
    monkeypatch.setattr(brain_ipc, 'BRAIN_IPC_ROOT', tmp_path / 'brain-ipc')


def start_worker(max_replans, max_calls):
    results = []
    worker = threading.Thread(target=lambda: results.append(brain_worker.worker_run(
        OfflineBrain(), max_calls=max_calls, max_seconds=3, max_replans=max_replans)))
    worker.start()
    until = time.monotonic() + 1
    while not brain_ipc.worker_status()['available'] and time.monotonic() < until:
        time.sleep(.002)
    assert brain_ipc.worker_status()['available']
    return worker, results


def test_eight_replans_survive_bridge_and_worker_without_reset(isolated_ipc):
    worker, results = start_worker(8, 10)
    try:
        bridge = brain_ipc.FileBrainBridge(max_replans=8)
        assert bridge.interpret('拿红块', SCENE).accepted
        assert bridge.status()['max_replans'] == 8
        for index in range(8):
            result = bridge.replan('继续', SCENE, PLAN, FEEDBACK)
            assert result.accepted, (index, result.reason)
        assert bridge.replan('继续', SCENE, PLAN, FEEDBACK).reason == 'replan_budget_exhausted'
        replacement = brain_ipc.FileBrainBridge(max_replans=8)
        assert replacement.replan('继续', SCENE, PLAN, FEEDBACK).reason == 'replan_budget_exhausted'
        assert replacement.interpret('拿红块', SCENE).accepted
    finally:
        worker.join(4)
    assert not worker.is_alive()
    assert results[0]['calls'] == 10 and results[0]['replans'] == 8
    assert results[0]['max_replans'] == 8


def test_legacy_bridge_default_one_even_with_eight_cap_worker(isolated_ipc):
    worker, results = start_worker(8, 3)
    try:
        bridge = brain_ipc.FileBrainBridge()
        assert bridge.max_replans == 1
        assert bridge.interpret('拿红块', SCENE).accepted
        assert bridge.replan('继续', SCENE, PLAN, FEEDBACK).accepted
        assert bridge.replan('继续', SCENE, PLAN, FEEDBACK).reason == 'replan_budget_exhausted'
        assert bridge.interpret('拿红块', SCENE).accepted
    finally:
        worker.join(4)
    assert not worker.is_alive() and results[0]['replans'] == 1


@pytest.mark.parametrize('cap', [-1, 9, True, 1.0, '8', None])
def test_invalid_caps_rejected_before_worker_store_access(cap):
    with pytest.raises(ValueError): brain_ipc.FileBrainBridge(max_replans=cap)
    with patch.object(brain_worker, 'BrainStore', side_effect=AssertionError('No storage should be opened')):
        with pytest.raises(ValueError): brain_worker.worker_run(OfflineBrain(), max_replans=cap)


@pytest.fixture
def locked_status(isolated_ipc):
    store = brain_ipc.BrainStore()
    fd = os.open('worker.lock', os.O_CREAT | os.O_RDWR, 0o600, dir_fd=store.fd)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def write(**changes):
        payload = {'status': 'ready', 'calls': 2, 'max_calls': 10,
                   'deadline_unix': time.time() + 30, 'replans': 0}
        payload.update(changes)
        store.write('ready.json', payload)
        return brain_ipc.worker_status()
    try:
        yield write
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        store.close()


def test_status_legacy_default_and_configured_eight(locked_status):
    legacy = locked_status()
    assert legacy['available'] and legacy['max_replans'] == legacy['remaining_replans'] == 1
    configured = locked_status(max_replans=8, replans=5)
    assert configured['available'] and configured['max_replans'] == 8
    assert configured['remaining_replans'] == 3


@pytest.mark.parametrize('changes', [{'max_replans': 9}, {'max_replans': -1}, {'max_replans': True},
    {'max_replans': '8'}, {'max_replans': 8.0}, {'max_replans': 8, 'replans': 9},
    {'max_replans': 8, 'replans': True}, {'max_replans': 0, 'replans': 1},
    {'replans': 2}, {'status': 'unknown'}])
def test_malformed_worker_status_fails_closed(locked_status, changes):
    status = locked_status(**changes)
    assert not status['available'] and status['remaining_replans'] == 0


@pytest.mark.parametrize('arguments,cap', [([], 1), (['--max-replans', '8'], 8)])
def test_cli_propagates_configured_replan_cap_without_loading_secrets(arguments, cap):
    fake = OfflineBrain()
    with patch('embodied_arm.providers.brain.QwenBrain.from_config', return_value=fake) as construct, \
            patch.object(brain_worker, 'worker_run', return_value={'status': 'mock'}) as run:
        brain_worker.main(['--config', '/nonexistent/mock-only', *arguments])
    assert construct.call_args.kwargs['max_replans'] == cap
    assert run.call_args.args == (fake, brain_ipc.MAX_CALLS, brain_ipc.MAX_SECONDS, cap)


def test_step_boundary_normalized_approach_is_valid_contract():
    context = {'cycle_index': 1, 'completed_micro_actions': 16, 'remaining_global_calls': 184,
               'held_object': 'red', 'held_verified': True, 'phase': 'approach',
               'residual_xyz_m': None, 'stop_reason': 'cycle_budget', 'observation_status': 'fresh',
               'recent_actions': []}
    assert validate_cycle_context(context) == context
    with pytest.raises(ValueError): validate_cycle_context({**context, 'phase': 'step_boundary'})
