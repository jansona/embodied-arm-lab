import copy
import hashlib
import json
import time
import pytest
from continue_final_stack import load_final_checkpoint,run_final
from test_continue_stack import fixture_files

def fixture(tmp_path):
    _,_,state,record=fixture_files(tmp_path)
    state.update(held_object='blue',ee=[.48,-.16,.27],gripper_width=.05,
                 finger_contacts_by_object={'blue':{'9':1,'10':1}})
    state['blocks']={'red':[.48,.21,.025],'blue':[.486,-.148,.265]}
    audit={'original_plan':record['plan'],'source_call_id':record['source_call_id'],'completed_prefix':['s1'],
      'remaining_step_results':[{'source_step_id':'s2','outcome':{'verified':True}}],
      'active_task':{'steps':[{'action':'pick','object_id':'blue','status':'completed','outcome':{'verified':True}}]}}
    prior=tmp_path/'prior.json';prior.write_text(json.dumps(audit))
    snapshot=tmp_path/'fresh.json';snapshot.write_text(json.dumps(state))
    p={k:record[k] for k in ('plan','source_call_id','model','provenance','accepted')}
    p.update(snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),completed_step_ids=['s1','s2','s3'],
             continuation_audit_sha256=hashlib.sha256(prior.read_bytes()).hexdigest())
    provenance=tmp_path/'p.json';provenance.write_text(json.dumps(p))
    return snapshot,provenance,prior,state,p

def test_exact_final_step_provenance_and_contacts(tmp_path):
    snapshot,p,prior,state,record=fixture(tmp_path)
    fresh,_=load_final_checkpoint(snapshot,p,prior)
    assert fresh['held_object']=='blue'
    prior.write_text(prior.read_text()+' ')
    with pytest.raises(ValueError):load_final_checkpoint(snapshot,p,prior)

def test_final_executes_only_fresh_place_and_measures(tmp_path):
    _,_,_,state,p=fixture(tmp_path);seen=[]
    class Backend:
        cancelled=False
        def __init__(self):self.data=copy.deepcopy(state)
        def state(self):return copy.deepcopy(self.data)
        def stop(self):pass
    backend=Backend()
    class Controller:
        calls=0;decisions=[]
        def cancel(self):pass
    class Executor:
        def __init__(self,backend,controller):pass
        def start(self,plan,**kwargs):
            seen.append(plan)
            backend.data['blocks']['blue']=[.48,.21,.075]
            backend.data.update(ee=[.48,.21,.27],gripper_width=.08,held_object=None)
        def status(self):return {'status':'completed'}
        def stop(self):pass
    result=run_final(state,p,backend,Controller(),time.monotonic()+10,tmp_path/'out.json',Executor)
    assert result['verified'] and result['new_qwen_calls']==0
    assert result['original_step']['step_id']=='s4'
    assert len(seen)==1 and seen[0]['steps'][0]['target_xyz']==state['ee']
    assert not any(x['action']=='close' for x in seen[0]['steps'])

def test_missing_grasp_contact_is_rejected_even_with_matching_hash(tmp_path):
    snapshot,p,prior,state,record=fixture(tmp_path)
    state['finger_contacts_by_object']['blue']['10']=0
    snapshot.write_text(json.dumps(state))
    record['snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    p.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='physical state'):
        load_final_checkpoint(snapshot,p,prior)
