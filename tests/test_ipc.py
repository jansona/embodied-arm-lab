"""Local mock-only tests; no credentials, provider requests, or robot actions."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from embodied_arm.providers.adapters import ACTIONS, Decision
import embodied_arm.runtime.ipc as ipc
from embodied_arm.runtime.ipc import FileBridgeAdapter, Store, normalize_state, worker_status
from embodied_arm.runtime.inference_worker import worker_run


def state():
    return {'schema':'arm-observation-v1','simulation_only':True,'robot':ipc.ROBOT,
            'task':ipc.TASK,'end_effector':[.4,0,.3],'joints':[0]*7,'gripper_width':.08,
            'objects':{'red':[.5,0,.025],'blue':[.58,-.19,.025]},'goal':[.48,.21,.025],
            'workspace':[[.25,.7],[-.35,.35],[.022,.6]],'busy':False,'paused':False,
            'metrics':{**{k:0 for k in ipc.METRIC_NUMBERS},**{k:False for k in ipc.METRIC_BOOLS},'finger_contacts':{'9':0,'10':0}},
            'capabilities':ipc.CAPABILITIES}


class MockAdapter:
    def __init__(self):self.calls=0
    def decide(self,state):
        self.calls+=1
        return Decision('x_plus',.99,{a:1.0 if a=='x_plus' else 0.0 for a in ACTIONS},True,'accepted')


class IPCTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/'.runtime'/'jev-ipc'
        self.patch=patch('embodied_arm.runtime.ipc.IPC_ROOT',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def job(self,job='a'*32,payload=None):
        store=Store()
        try:store.write(job+'.job',{'id':job,'expires_at':time.time()+10,'state':state()} if payload is None else payload)
        finally:store.close()
        return job
    def test_roundtrip_and_counter_bounds(self):
        adapter=MockAdapter();done=[]
        thread=threading.Thread(target=lambda:done.append(worker_run(adapter,max_calls=1,max_seconds=1)))
        thread.start()
        result=FileBridgeAdapter().decide(state());thread.join(2)
        self.assertTrue(result.accepted);self.assertEqual(result.action,'x_plus')
        self.assertEqual(adapter.calls,1);self.assertEqual(done[0]['calls'],1)
        self.assertEqual(done[0]['status'],'call_budget')
        self.assertFalse(worker_status()['available'])
        self.assertEqual(self.root.stat().st_mode&0o777,0o700)
    def test_objective_validation(self):
        hover=state();hover['task']=ipc.HOVER_TASK
        hover['objective']={'id':'hover_reach','target_xyz':[.5,0,.44],'tolerance_m':.01}
        self.assertEqual(normalize_state(hover)['objective']['id'],'hover_reach')
        hover['objective']['target_xyz'][2]=.025
        with self.assertRaises(ValueError):normalize_state(hover)
        local=state();local['task']=ipc.LOCAL_TASK
        local['objective']={'id':'local_target','target_xyz':[.45,0,.3],'tolerance_m':.01}
        self.assertEqual(normalize_state(local)['objective']['id'],'local_target')
        local['objective']['target_xyz'][0]=20
        with self.assertRaises(ValueError):normalize_state(local)
        pick=state();pick['objective']={'id':'pick_place','target_xyz':pick['goal'],'tolerance_m':.055}
        self.assertEqual(normalize_state(pick)['objective']['id'],'pick_place')
        pick['objective']['target_xyz']=[0,0,0]
        with self.assertRaises(ValueError):normalize_state(pick)

    def test_low_confidence_audit_retains_proposal_and_safe_state(self):
        probabilities={a:1.0 if a=='y_plus' else 0.0 for a in ACTIONS}
        class Low:
            def decide(self,state):return Decision(confidence=.35,probabilities=probabilities,reason='low_confidence',proposed_action='y_plus')
        self.job();result=worker_run(Low(),max_calls=1,max_seconds=1)
        store=Store()
        try:audit=store.read('last-run-audit.json')
        finally:store.close()
        self.assertEqual(result['calls'],1);self.assertEqual(len(audit),1)
        self.assertEqual(audit[0]['decision']['action'],'hold')
        self.assertEqual(audit[0]['decision']['proposed_action'],'y_plus')
        self.assertEqual(audit[0]['decision']['confidence'],.35)
        self.assertEqual(audit[0]['state'],state())
        self.assertGreaterEqual(audit[0]['latency_seconds'],0)
        self.assertNotIn('Authorization',json.dumps(audit))
        worker_run(Low(),max_seconds=.03)
        store=Store()
        try:self.assertEqual(store.read('last-run-audit.json'),[])
        finally:store.close()

    def test_eight_audit_timings_stay_bounded_and_sanitized(self):
        class Traced(MockAdapter):
            last_transport={'request_start':0,'request_body_sent':.1,'elapsed_seconds':.2,
                            'headers':{'Authorization':'SECRET'},'error_type':{'SECRET':1},
                            'events':[{'name':'http11.receive_response_headers.started','seconds':.1,'info':'SECRET'}]*30}
        for index in range(8):self.job(format(index,'032x'))
        result=worker_run(Traced(),max_calls=8,max_seconds=1)
        self.assertEqual(result['calls'],8)
        path=self.root/'last-run-audit.json'
        self.assertLess(path.stat().st_size,32768)
        content=path.read_text();self.assertNotIn('SECRET',content)
        audit=json.loads(content);self.assertEqual(len(audit),8)
        self.assertEqual(len(audit[0]['transport']['events']),12)

    def test_cancel_before_worker_means_no_call(self):
        bridge=FileBridgeAdapter();result=[]
        thread=threading.Thread(target=lambda:result.append(bridge.decide(state())))
        thread.start()
        deadline=time.monotonic()+1
        while not list(self.root.glob('*.job')) and time.monotonic()<deadline:time.sleep(.005)
        bridge.cancel();thread.join(1)
        adapter=MockAdapter();worker_run(adapter,max_calls=1,max_seconds=.05)
        self.assertEqual(adapter.calls,0);self.assertEqual(result[0].reason,'cancelled')
    def test_cancel_pending_result_never_returned(self):
        entered,release=threading.Event(),threading.Event()
        class Pending(MockAdapter):
            def decide(self,state):entered.set();release.wait(1);return super().decide(state)
        adapter=Pending();worker=threading.Thread(target=lambda:worker_run(adapter,max_calls=1,max_seconds=1));worker.start()
        bridge=FileBridgeAdapter();results=[]
        client=threading.Thread(target=lambda:results.append(bridge.decide(state())));client.start()
        self.assertTrue(entered.wait(1));bridge.cancel();release.set();client.join(1);worker.join(1)
        self.assertEqual(results[0].reason,'cancelled')
    def test_strict_schema_and_malformed_jobs_never_call(self):
        bad=state();bad['headers']={'Authorization':'do not forward'}
        self.assertEqual(FileBridgeAdapter().decide(bad).reason,'invalid_state')
        self.job(payload={'id':'a'*32,'expires_at':time.time()+10,'state':bad})
        adapter=MockAdapter();result=worker_run(adapter,max_seconds=.06)
        self.assertEqual(adapter.calls,0);self.assertEqual(result['rejected'],1)
        for mutate in [lambda s:s.update(task='Arbitrary proxy request'),lambda s:s.update(end_effector=[float('nan'),0,0]),lambda s:s.update(objects={'red':[0,0,0]})]:
            bad=state();mutate(bad)
            with self.assertRaises(ValueError):normalize_state(bad)
    def test_symlink_and_path_traversal_rejected(self):
        store=Store()
        try:
            with self.assertRaises(ValueError):store.read('../secrets')
            target=Path(self.temp.name)/'target';target.write_text('{}')
            (self.root/('a'*32+'.job')).symlink_to(target)
            with self.assertRaises(OSError):store.read('a'*32+'.job')
            with self.assertRaises(ValueError):store.write('a'*32+'.job',{})
        finally:store.close()
        adapter=MockAdapter();result=worker_run(adapter,max_seconds=.05)
        self.assertEqual(adapter.calls,0);self.assertEqual(result['rejected'],1)
        alternate=Path(self.temp.name)/'link';alternate.symlink_to(self.root,target_is_directory=True)
        with patch('embodied_arm.runtime.ipc.IPC_ROOT',alternate),self.assertRaises(ValueError):Store()
    def test_worker_window_and_ready_status(self):
        adapter=MockAdapter();results=[]
        thread=threading.Thread(target=lambda:results.append(worker_run(adapter,max_seconds=.15)))
        thread.start();deadline=time.monotonic()+1
        # Do not probe the same flock before the worker has acquired it: the
        # status probe briefly owns an unlocked file and can race startup.
        while not (self.root/'ready.json').exists() and time.monotonic()<deadline:time.sleep(.005)
        while not worker_status()['available'] and time.monotonic()<deadline:time.sleep(.005)
        self.assertTrue(worker_status()['available'])
        with self.assertRaises(ValueError):worker_run(adapter,max_seconds=.01)
        thread.join(1)
        self.assertFalse(worker_status()['available']);self.assertEqual(results[0]['status'],'time_budget')
        for kwargs in [{'max_calls':9},{'max_calls':True},{'max_seconds':31},{'max_seconds':0}]:
            with self.assertRaises(ValueError):worker_run(adapter,**kwargs)
    def test_bridge_timeout_without_waiting_ten_seconds(self):
        real=time.monotonic;started=real()
        def accelerated():return started+(real()-started)*1000
        with patch('embodied_arm.runtime.ipc.time.monotonic',side_effect=accelerated):
            result=FileBridgeAdapter().decide(state())
        self.assertEqual(result.reason,'timeout');self.assertTrue(list(self.root.glob('*.cancel')))
    def test_pending_request_does_not_extend_worker_deadline(self):
        release=threading.Event()
        class Slow:
            def decide(self,state):release.wait(1);return Decision(reason='timeout')
        self.job();before=time.monotonic()
        result=worker_run(Slow(),max_seconds=.05)
        release.set()
        self.assertLess(time.monotonic()-before,.2);self.assertEqual(result['calls'],1)
        self.assertEqual(result['status'],'time_budget')


if __name__=='__main__':unittest.main()
