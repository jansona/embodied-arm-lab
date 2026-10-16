"""One explicit API-only seven-case direction/hold audit. Never actuates a robot.

The request contains synthetic coordinates and neutral case labels only. Expected
answers are generated locally for scoring, never included in the request.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import httpx
from embodied_arm.providers.adapters import ACTIONS, CRITERIA, JevAdapter, MAX_RESPONSE_BYTES, validate_provider


def coordinate_cases():
    """Deterministic axis-aligned test observations, with no action answers."""
    current=[.45,0.0,.30]
    cases={}
    for label,(axis,sign) in zip('ABCDEF',[(0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)]):
        target=current.copy();target[axis]=round(target[axis]+sign*.06,4)
        cases[label]={'current_xyz_m':current.copy(),'target_xyz_m':target}
    cases['G']={'current_xyz_m':current.copy(),'target_xyz_m':current.copy()}
    return cases


def expected_direction(case):
    """Offline scoring only; this function never controls a physical action."""
    delta=[round(target-current,6) for current,target in zip(case['current_xyz_m'],case['target_xyz_m'])]
    if all(abs(value)<=.01 for value in delta):return 'hold'
    nonzero=[index for index,value in enumerate(delta) if abs(value)>.01]
    if len(nonzero)!=1:raise ValueError('Audit requires one changed axis')
    axis=nonzero[0]
    return 'xyz'[axis]+('_plus' if delta[axis]>0 else '_minus')


def batch_request(adapter):
    validate_provider(adapter.endpoint,adapter.model)
    cases=coordinate_cases()
    return {'model':adapter.model,'state':{'cases':cases,'step_m':.02,'tolerance_m':.01},'questions':{
        label:{'type':'choice',
               'instructions':f'For case {label} in state.cases, compare current_xyz_m and target_xyz_m. Choose hold if every absolute coordinate error is within state.tolerance_m; otherwise choose one 2 cm axis jog that reduces distance to that target. Positive X, Y, or Z increases that coordinate; negative decreases it. Judge this case independently. Do not open or close the gripper.',
               'criteria':dict(CRITERIA)} for label in cases}}


def run_audit(adapter):
    """Send exactly one seven-question request using the configured app adapter."""
    if not isinstance(adapter,JevAdapter):raise ValueError('A configured Jev adapter is required')
    body=batch_request(adapter)
    trace,mark=adapter._begin_transport()
    result={'status':'invalid_response','http_status':None,'model':adapter.model,
            'cases':[],'correct':0,'total':7,'accuracy':0.0,'accepted':0,'transport':{}}
    try:
        with httpx.Client(timeout=httpx.Timeout(adapter.timeout),follow_redirects=False,
                          trust_env=adapter.use_environment_network,transport=adapter._transport) as client:
            with client.stream('POST',adapter.endpoint,
                               headers={'Authorization':f'Bearer {adapter._api_key}','Accept':'application/json'},
                               json=body,extensions={'trace':trace}) as response:
                mark('response_headers_received');result['http_status']=response.status_code
                if response.status_code!=200:
                    result['status']='http_error';return result
                raw=bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw)>MAX_RESPONSE_BYTES:
                        result['status']='response_too_large';return result
        mark('response_complete')
        payload=json.loads(raw);adapter._server_latency(payload)
        answers=payload.get('answers') if isinstance(payload,dict) else None
        result['schema_audit']={'answers_is_object':isinstance(answers,dict),'missing_cases':[c for c in 'ABCDEFG' if not isinstance(answers,dict) or c not in answers], 'case_details':{}}
        if isinstance(answers,dict):
            for label in 'ABCDEFG':
                a=answers.get(label);d={'answer_is_object':isinstance(a,dict)}
                if isinstance(a,dict):
                    d['type_is_choice']=a.get('type')=='choice'
                    d['choice']=a.get('choice') if isinstance(a.get('choice'),str) and a.get('choice') in ACTIONS else None
                    c=a.get('confidence');d['confidence']=c if type(c) in (int,float) and math.isfinite(c) and 0<=c<=1 else None
                    p=a.get('probabilities');d['probabilities']={k:v for k,v in p.items() if k in ACTIONS and type(v) in (int,float) and math.isfinite(v) and 0<=v<=1} if isinstance(p,dict) else {}
                    d['probability_sum']=sum(d['probabilities'].values())
                    d['validation_reason']=adapter._validate({'answers':{'action':a}}).reason
                result['schema_audit']['case_details'][label]=d
        if not isinstance(payload,dict) or not isinstance(payload.get('answers'),dict) or set(payload['answers'])!=set('ABCDEFG'):
            return result
        rows=[]
        for label,case in coordinate_cases().items():
            decision=adapter._validate({'answers':{'action':payload['answers'][label]}})
            if decision.reason not in ('accepted','low_confidence'):return result
            expected=expected_direction(case)
            actual=decision.proposed_action
            rows.append({'case':label,'current_xyz_m':case['current_xyz_m'],'target_xyz_m':case['target_xyz_m'],
                         'expected':expected,'actual':actual,'correct':actual==expected,
                         'confidence':decision.confidence,'probabilities':decision.probabilities,
                         'accepted':decision.accepted,'reason':decision.reason})
        correct=sum(row['correct'] for row in rows)
        result.update(status='complete',cases=rows,correct=correct,accuracy=correct/len(rows),
                      accepted=sum(row['accepted'] for row in rows))
    except httpx.HTTPError as error:
        adapter._transport_error(error)
        result['status']='timeout' if isinstance(error,httpx.TimeoutException) else 'network_error'
    except (ValueError,UnicodeError,RecursionError):
        result['status']='invalid_response'
    finally:
        mark('elapsed_seconds');result['transport']=adapter.last_transport
    return result


class NoRobot:
    def stop(self):pass


def main():
    parser=argparse.ArgumentParser(description='One seven-case direction/hold API-only audit; never moves the arm')
    parser.add_argument('--environment-network',action='store_true')
    args=parser.parse_args()
    from embodied_arm.legacy.panda.runner import JevRunner
    runner=JevRunner(NoRobot(),use_environment_network=args.environment_network)
    try:
        runner.configure(Path(__file__).resolve().parents[2]/'jev-arm-config.json')
        if runner._adapter is None:raise ValueError('Not configured')
        print(json.dumps(run_audit(runner._adapter),indent=2))
    except Exception:
        print(json.dumps({'status':'audit_error'}))
        raise SystemExit(1) from None


if __name__=='__main__':main()
