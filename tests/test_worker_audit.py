import json
import pytest
import threading
from pathlib import Path
import time
from adapters import ACTIONS,Decision
from inference_worker import AuditRecorder,worker_run
from ipc import Store,FileBridgeAdapter
from test_ipc import state

def test_audit_more_than_48_records_is_bounded(tmp_path):
    store=Store(tmp_path/'ipc');audit=AuditRecorder(store)
    try:
        entry={'decision':Decision('x_plus',.99,{a:float(a=='x_plus') for a in ACTIONS},True,'accepted','x_plus').to_dict(),'state':state(),'transport':{'test_padding':'x'*1500},'latency_seconds':.5,'delivered':True}
        for _ in range(200):audit.record(entry)
        assert len(store.read('last-run-audit.json'))==8
        assert len(store.read('run-summary.json'))==128
        assert audit.errors==0
    finally:store.close()

def test_audit_write_failure_is_nonfatal():
    class Broken:
        def write(self,*args):raise OSError('test unavailable')
    audit=AuditRecorder(Broken())
    audit.record({'decision':Decision().to_dict(),'latency_seconds':0,'delivered':True})
    assert audit.errors==1

@pytest.mark.parametrize('budget',[48,120,130,200])
def test_full_bounded_mock_ipc_calls(tmp_path,monkeypatch,budget):
    import ipc
    monkeypatch.setattr(ipc,'IPC_ROOT',tmp_path/'ipc')
    class Model:
        max_worker_calls=200;max_worker_seconds=420
        def decide(self,observation):return Decision('x_plus',.99,{a:float(a=='x_plus') for a in ACTIONS},True,'accepted','x_plus')
    result=[];thread=threading.Thread(target=lambda:result.append(worker_run(Model(),budget,8)));thread.start()
    bridge=FileBridgeAdapter()
    for _ in range(budget):assert bridge.decide(state()).accepted
    thread.join(10)
    assert not thread.is_alive()
    assert result[0]['calls']==budget and result[0]['processed']==budget
    assert result[0]['audit_write_errors']==0
    store=Store()
    try:assert len(store.read('run-summary.json'))==min(budget,128)
    finally:store.close()
