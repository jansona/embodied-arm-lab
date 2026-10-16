"""Stop/re-observe integration with synthetic RGB and robot-only fixtures."""
from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from embodied_arm.control.manipulation import VisualManipulationExecutor
from tests.test_held_continuity import certificate, snapshot
from tests.test_jev_motion import FakeSim


class ReacquiringObserver:
    validated = True

    def __init__(self, sim, now):
        self.sim, self.now = sim, now
        self.cancel_count = 0
        self.adopted = []
        self.prepare_count = 0
        self.prepare_callback = None
        self.adopt_callback = None
        self.change_fresh = None

    def cancel(self):
        self.cancel_count += 1

    def prepare_hold_reacquisition(self, object_id):
        self.prepare_count += 1
        fresh = snapshot(self.now[0], epoch=self.sim.state['epoch'])
        fresh['robot'] = deepcopy(self.sim.state)
        if self.change_fresh:
            self.change_fresh(fresh)
        if self.prepare_callback:
            self.prepare_callback()
        return fresh

    def adopt_verified_hold(self, proof):
        self.adopted.append(deepcopy(proof))
        if self.adopt_callback:
            self.adopt_callback()


def fixture(*, held=True):
    now = [2.]
    sim = FakeSim()
    old = snapshot(1., epoch=1)
    sim.state = deepcopy(old['robot'])
    observer = ReacquiringObserver(sim, now)
    executor = VisualManipulationExecutor(simulation=sim, observer=observer,
        driver=SimpleNamespace(validated=True), clock=lambda: now[0])
    if held:
        executor.held = 'red'
        executor.held_evidence = certificate(old)
        executor._verified_lift_snapshot = deepcopy(old)
        executor.holding_status = 'verified_held'
    return sim, observer, executor, now


def test_stop_retains_only_historical_candidate_and_never_moves_or_opens_gripper():
    sim, observer, executor, _ = fixture()
    before = deepcopy(sim.state)
    assert executor.planning_held_object() == 'red'
    executor.stop()
    assert executor.held == 'red'  # Historical identity is not a live claim.
    assert executor.held_evidence is None
    assert executor._historical_held_candidate is not None
    assert executor.planning_held_object() is None
    assert executor.robot_state() == {'joints': [0.] * 5, 'gripper_openness': .2,
                                      'holding_status': 'unknown'}
    assert sim.state['gripper_openness'] == before['gripper_openness']
    assert sim.state['gripper_angle'] == before['gripper_angle']
    assert sim.state['ee'] == before['ee'] and sim.state['joints'] == before['joints']
    assert sim.state['actions'] == before['actions'] and sim.commands == []
    assert executor._stopped_robot['epoch'] == sim.state['epoch'] == 2
    assert executor._stopped_at == 2. and observer.cancel_count == 1


def test_new_command_reacquires_visible_held_object_without_any_probe_motion():
    sim, observer, executor, now = fixture()
    executor.stop()
    now[0] = 3.
    result = executor.reconcile_observation()
    assert result['verified'], result
    assert executor.planning_held_object() == 'red'
    assert executor.robot_state()['holding_status'] == 'verified_held'
    assert observer.adopted == [result['certificate']]
    assert executor.held_evidence['epoch'] == 2
    assert executor.held_evidence['frame_ids'] == {'left': 'left-3.0', 'right': 'right-3.0'}
    assert executor._historical_held_candidate is None
    assert executor.reconcile_observation() is None
    assert sim.commands == []


def test_repeated_stop_and_cancellation_epoch_do_not_erase_unchanged_history():
    sim, _, executor, now = fixture()
    executor.stop()
    candidate = executor._historical_held_candidate
    sim.stop()  # A driver's cancellation cleanup can issue its own Stop.
    now[0] = 2.5
    executor.stop()
    assert executor._historical_held_candidate is candidate
    assert executor._stopped_robot['epoch'] == 4
    now[0] = 3.
    assert executor.reconcile_observation()['verified']


@pytest.mark.parametrize('field,value', [('actions', 13), ('joints', [.1, 0., 0., 0., 0.]),
    ('ee', [.22, 0., .07]), ('gripper_openness', .8), ('gripper_angle', 1.)])
def test_repeated_stop_cannot_hide_intervening_movement_or_gripper_change(field, value):
    sim, observer, executor, _ = fixture()
    executor.stop()
    sim.state[field] = value
    executor.stop()
    assert executor._historical_held_candidate is None
    assert executor.reconcile_observation() is None
    assert executor.holding_status == 'unknown' and observer.prepare_count == 0


@pytest.mark.parametrize('phase,state', [('close', 'unknown'), ('closing', 'held_verified'),
    ('release', 'held_verified'), ('released', 'release_pending'),
    ('verify_release', 'held_verified'), ('transit', 'unknown')])
def test_stop_never_retains_prior_hold_after_release_or_uncertain_grasp(phase, state):
    _, observer, executor, _ = fixture()
    executor._run_state = {'phase': phase, 'holding_state': state, 'pending': None}
    executor.stop()
    assert executor._historical_held_candidate is None
    assert executor.reconcile_observation() is None
    assert executor.robot_state()['holding_status'] == 'unknown'
    assert observer.prepare_count == 0


def test_pending_release_is_rejected_even_before_phase_state_advances():
    _, _, executor, _ = fixture()
    executor._run_state = {'phase': 'lower', 'holding_state': 'held_verified',
                           'pending': {'phase': 'release', 'kind': 'gripper', 'target': 1}}
    executor.stop()
    assert executor._historical_held_candidate is None


def test_validated_empty_stop_before_close_stays_empty_and_reset_clears_all_history():
    _, _, executor, _ = fixture(held=False)
    executor._run_state = {'phase': 'approach', 'holding_state': 'empty', 'pending': None}
    executor.stop()
    assert executor.robot_state()['holding_status'] == 'verified_empty'
    _, _, held_executor, _ = fixture()
    held_executor.stop()
    held_executor.reset()
    assert held_executor.held is None and held_executor.held_evidence is None
    assert held_executor._historical_held_candidate is None
    assert held_executor._verified_lift_snapshot is None
    assert held_executor._stopped_robot is None
    assert held_executor.holding_status == 'verified_empty'


def test_ambiguous_fresh_scene_keeps_unknown_without_adopting_old_proof():
    _, observer, executor, now = fixture()
    executor.stop()
    now[0] = 3.
    observer.change_fresh = lambda fresh: fresh['visual_evidence'].update(association_candidates=2)
    result = executor.reconcile_observation()
    assert not result['verified'] and result['certificate'] is None
    assert executor.held_evidence is None and executor.holding_status == 'unknown'
    assert observer.adopted == []
    assert executor.planning_held_object() is None


@pytest.mark.parametrize('when', ['prepare', 'adopt'])
def test_stop_during_reacquisition_cannot_commit_the_inflight_certificate(when):
    _, observer, executor, now = fixture()
    executor.stop()
    now[0] = 3.
    setattr(observer, f'{when}_callback', executor.stop)
    result = executor.reconcile_observation()
    assert not result['verified'] and executor.held_evidence is None
    assert executor.holding_status == 'unknown'
    assert executor.planning_held_object() is None


def test_unknown_holding_blocks_new_pick_instead_of_assuming_empty():
    _, _, executor, _ = fixture()
    executor.stop()
    with pytest.raises(ValueError, match='holding_state_unknown'):
        executor.execute({'action': 'pick', 'object_id': 'blue', 'target_id': None})


def test_closing_becomes_unknown_before_driver_can_be_stopped():
    _, _, executor, _ = fixture(held=False)
    def program(plan):
        yield {'kind': 'gripper', 'phase': 'close', 'target': 0, 'geometry': {}}
        return {'verified': True}
    executor._program = program
    def gripper(*args, **kwargs):
        assert executor.holding_status == 'unknown'
        executor.stop()
        return {'status': 'stopped', 'reason': 'perception_rejected', 'calls': 1, 'actions': 0}
    executor.driver.gripper = gripper
    result = executor.execute({'action': 'pick', 'object_id': 'red', 'target_id': None})
    assert not result['verified'] and executor.holding_status == 'unknown'
    assert executor._historical_held_candidate is None


def test_external_stop_cancellation_unwind_preserves_historical_verified_candidate():
    _, _, executor, now = fixture()
    cancel = threading.Event()
    def program(plan):
        yield {'kind': 'move', 'phase': 'transit', 'target': [.2, 0., .07],
               'covariance': [[0.] * 3] * 3}
        return {'verified': True}
    executor._program = program
    def move(*args, **kwargs):
        executor.stop();cancel.set()
        return {'status': 'stopped', 'reason': 'cancelled', 'calls': 1, 'actions': 0}
    executor.driver.move_to = move
    result = executor.execute({'action': 'place', 'object_id': 'red', 'target_id': 'goal'},
                              cancel_event=cancel)
    assert result['status'] == 'cancelled'
    assert executor._historical_held_candidate is not None
    now[0] = 3.
    assert executor.reconcile_observation()['verified']


def test_genuine_execution_failure_discards_old_hold_and_keeps_typed_rgb_diagnostics():
    _, observer, executor, _ = fixture()
    observer.last_failure_details = {'reason': 'rgb_fit_rejected', 'views': []}
    observer.last_error = 'retry RGB error'
    original_details = observer.last_failure_details
    def clear_diagnostics():
        observer.last_failure_details = None
        observer.last_error = None
    observer.cancel = clear_diagnostics
    def program(plan):
        yield {'kind': 'move', 'phase': 'transit', 'target': [.2, 0., .07],
               'covariance': [[0.] * 3] * 3}
        return {'verified': True}
    executor._program = program
    executor.driver.move_to = lambda *a, **k: {'status': 'stopped', 'reason': 'perception_rejected',
                                               'calls': 1, 'actions': 0}
    result = executor.execute({'action': 'place', 'object_id': 'red', 'target_id': 'goal'})
    assert not result['verified'] and result['holding_state'] == 'unknown'
    assert executor.holding_status == 'unknown'
    assert executor._historical_held_candidate is None and executor.held_evidence is None
    assert observer.last_failure_details is None and observer.last_error is None
    assert result['perception_evidence'] == original_details
    assert result['perception_detail'] == result['recovery_detail'] == 'retry RGB error'
    original_details['views'].append('changed')
    assert result['perception_evidence']['views'] == []


@pytest.mark.parametrize('action,verification', [('pick', 'verify_lift'), ('place', 'verify_release')])
def test_stop_during_final_verification_never_commits_a_late_held_or_empty_proof(action, verification):
    sim, observer, executor, now = fixture(held=action == 'place')
    sim.grasp_geometry = lambda target: {'approach_offset_world': [0., 0., 0.], 'tcp_below_top_m': .007}
    observer.configure_task = lambda action: None
    observer.begin_step = lambda *a, **k: None
    observer.begin_phase = lambda *a: snapshot(now[0], epoch=sim.state['epoch'])
    observer.fixed_goal = lambda: {'estimate': snapshot(now[0])['scene']['objects']['goal']}
    def late_verification(*args):
        executor.stop()
        return {'verified': True}  # Simulates in-flight work completing after Stop.
    setattr(observer, verification, late_verification)
    executor.driver.gripper = executor.driver.move_to = lambda *a, **k: {
        'status': 'completed', 'calls': 1, 'actions': 0, 'provenance': 'offline_fixture'}
    result = executor.execute({'action': action, 'object_id': 'red',
                               'target_id': 'goal' if action == 'place' else None})
    assert not result['verified'], result
    assert executor.holding_status == 'unknown'
    assert executor.held_evidence is None
    assert executor._verified_lift_snapshot is None
    assert executor._historical_held_candidate is None
    assert executor.planning_held_object() is None


def test_verified_lift_commits_a_copied_original_snapshot_for_future_stop():
    sim, observer, executor, now = fixture(held=False)
    original = snapshot(now[0], epoch=sim.state['epoch'])
    sim.grasp_geometry = lambda target: {'approach_offset_world': [0., 0., 0.], 'tcp_below_top_m': .007}
    observer.configure_task = lambda action: None
    observer.begin_step = lambda *a, **k: None
    observer.begin_phase = lambda *a: original
    observer.verify_lift = lambda *a: {'verified': True}
    executor.driver.gripper = executor.driver.move_to = lambda *a, **k: {
        'status': 'completed', 'calls': 1, 'actions': 0, 'provenance': 'offline_fixture'}
    result = executor.execute({'action': 'pick', 'object_id': 'red', 'target_id': None})
    assert result['verified'] and executor.holding_status == 'verified_held'
    assert executor._verified_lift_snapshot == original
    original['scene']['frame_ids']['left'] = 'mutated-after-verification'
    assert executor._verified_lift_snapshot['scene']['frame_ids']['left'] == 'left-2.0'
    assert executor.held_evidence['frame_ids']['left'] == 'left-2.0'
