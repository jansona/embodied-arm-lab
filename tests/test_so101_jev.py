"""SO101-only mock tests. No simulator oracle, configuration, or live API."""
import copy
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import httpx
from embodied_arm.providers.adapters import ALIYUN_ENDPOINT, ALIYUN_MODEL
from embodied_arm.providers.so101_jev import (SO101JevAdapter,SO101Decision,build_observation,request_preview,
                       validate_observation,parse_response,validate_decision)


def robot():
    return {'name':'SO-ARM101','arm_dof':5,'ee':[.2,0,.06],'joints':[0]*5,'joint_limits':[[-2,2]]*5,
            'gripper_openness':.4,'gripper_angle':.2,'paused':False,'busy':False,'epoch':1,'actions':0,
            'workspace':[[.06,.38],[-.25,.25],[.006,.36]],'cube_edge_m':.02}


def observation(axis=0,delta=.02):
    current=robot();target=current['ee'].copy();target[axis]+=delta
    return build_observation({'source':'rgb_only','kind':'cartesian','target_xyz':target,'tolerance_m':.005,'step_m':.01},current,axis)


def gripper(openness=.4,target=1):
    current=robot();current['gripper_openness']=openness
    return build_observation({'source':'rgb_only','kind':'gripper','target_openness':target,'tolerance':.02},current)


def answer(choice='positive',confidence=.9,grip=False):
    candidates=('open','close','hold') if grip else ('positive','negative','hold')
    return {'model':ALIYUN_MODEL,'answers':{'action':{'type':'choice','choice':choice,'confidence':confidence,
            'probabilities':{a:.9 if a==choice else .05 for a in candidates}}}}


class SO101JevTests(unittest.TestCase):
    def adapter(self,payload=None,handler=None,**kwargs):
        handler=handler or (lambda request:httpx.Response(200,json=payload))
        return SO101JevAdapter(api_key='mock-only-placeholder',transport=httpx.MockTransport(handler),**kwargs)
    def test_request_is_so101_rgb_target_plus_encoders_only(self):
        obs=observation(1,-.02);preview=request_preview(obs);state=preview['body']['state']
        self.assertEqual(preview['endpoint'],ALIYUN_ENDPOINT);self.assertEqual(preview['body']['model'],ALIYUN_MODEL)
        self.assertEqual(state['proprioception']['name'],'SO-ARM101');self.assertEqual(state['proprioception']['arm_dof'],5)
        self.assertEqual(state['signed_error_m'],-.02);self.assertEqual(state['axis_index'],1)
        self.assertIn('does not select direction',state['axis_schedule'])
        self.assertEqual(set(preview['body']['questions']['action']['criteria']),{'positive','negative','hold'})
        for forbidden in ['Panda','body_id','evaluation','gripper_tactile','contact_count','chosen_action','selected_action','measured_target_side','measured_distance_band']:
            self.assertNotIn(forbidden,json.dumps(preview))
        state['proprioception']['ee'][0]=9
        self.assertEqual(obs['robot']['ee'][0],.2)
    def test_axis_schedule_is_supplied_not_selected_by_error(self):
        obs=observation(0,0);obs['target']['target_xyz'][1]=.1
        state=request_preview(obs)['body']['state']
        self.assertEqual(state['axis_name'],'x');self.assertEqual(state['signed_error_m'],0)
        result=parse_response(answer('hold'),obs)
        self.assertEqual(result.action,'hold');self.assertTrue(result.accepted)
    def test_all_cartesian_directions_and_hold(self):
        for axis in range(3):
            for delta,choice,suffix in [(.02,'positive','_plus'),(-.02,'negative','_minus'),(0,'hold',None)]:
                with self.subTest(axis=axis,delta=delta):
                    result=parse_response(answer(choice),observation(axis,delta))
                    self.assertTrue(result.accepted)
                    self.assertEqual(result.action,'hold' if suffix is None else 'xyz'[axis]+suffix)
                    self.assertEqual(validate_decision(result.to_dict(),observation(axis,delta)),result)
    def test_exclusive_inclusive_hold_boundaries(self):
        for delta in [-.005,0,.005]:
            obs=observation(0,delta)
            self.assertTrue(parse_response(answer('hold'),obs).accepted)
            result=parse_response(answer('positive' if delta>=0 else 'negative'),obs)
            self.assertFalse(result.accepted);self.assertIsNone(result.action)
            self.assertEqual(result.reason,'criterion_mismatch')
        self.assertTrue(parse_response(answer('positive'),observation(0,.0051)).accepted)
        self.assertTrue(parse_response(answer('negative'),observation(0,-.0051)).accepted)
    def test_low_confidence_and_wrong_direction_have_no_fallback(self):
        low=parse_response(answer(confidence=.649),observation())
        self.assertFalse(low.accepted);self.assertIsNone(low.action);self.assertEqual(low.proposed_choice,'positive')
        wrong=parse_response(answer('negative'),observation())
        self.assertFalse(wrong.accepted);self.assertIsNone(wrong.action);self.assertEqual(wrong.proposed_choice,'negative')
        self.assertEqual(wrong.reason,'criterion_mismatch')
        self.assertTrue(parse_response(answer(confidence=.65),observation()).accepted)
    def test_gripper_endpoints_no_partial_servo_or_tactile(self):
        for current,target,choice in [(.4,1,'open'),(.4,0,'close'),(.99,1,'hold'),(.01,0,'hold')]:
            obs=gripper(current,target);result=parse_response(answer(choice,grip=True),obs)
            self.assertTrue(result.accepted);self.assertEqual(result.action,choice)
            state=request_preview(obs)['body']['state']
            self.assertNotIn('step',state);self.assertNotIn('contact_count',json.dumps(state))
            self.assertIn('does not prove a grasp',state['grasp_verification'])
        with self.assertRaises(ValueError):gripper(target=.5)
        obs=gripper();obs['target']['step']=.1
        with self.assertRaises(ValueError):validate_observation(obs)
    def test_strict_input_boundary_rejects_oracle_and_invalid_values(self):
        mutations=[lambda x:x['robot'].update(body_id=7),lambda x:x['robot'].update(gripper_tactile={}),
                   lambda x:x.update(evaluation={'success':True}),lambda x:x['target'].update(chosen_action='x_plus'),
                   lambda x:x['target'].update(source='simulator'),lambda x:x['robot'].update(name='Franka Panda'),
                   lambda x:x['robot'].update(arm_dof=7),lambda x:x['robot'].update(ee=[float('nan'),0,0]),
                   lambda x:x['robot'].update(gripper_openness=True),lambda x:x.update(axis_index=True),
                   lambda x:x['target'].update(target_xyz=[999,0,0]),lambda x:x['target'].update(step_m=.1)]
        for mutate in mutations:
            obs=observation();mutate(obs)
            with self.subTest(obs=obs),self.assertRaises(ValueError):validate_observation(obs)
        obs=observation();obs['target']['tolerance_m']=float('inf')
        with self.assertRaises(ValueError):validate_observation(obs)
    def test_malformed_response_rejected(self):
        cases=[None,[],{}, {'answers':{'action':{}}}]
        for key,value in [('type','score'),('choice','x_plus'),('choice',[]),('confidence',True),('confidence',float('nan')),
                          ('probabilities',{'positive':1}),('confidence',-1)]:
            data=answer();data['answers']['action'][key]=value;cases.append(data)
        bad=answer();bad['answers']['action']['probabilities']['negative']=.9;cases.append(bad)
        bad=answer();bad['model']='other-provider-model';cases.append(bad)
        bad=answer();del bad['model'];cases.append(bad)
        for data in cases:
            result=parse_response(data,observation())
            self.assertFalse(result.accepted);self.assertIsNone(result.action);self.assertEqual(result.reason,'invalid_response')
    def test_rounding_normalizes_only_bounded_three_way_distribution(self):
        data=answer();data['answers']['action']['probabilities']={'positive':.89,'negative':.05,'hold':.05}
        result=parse_response(data,observation());self.assertTrue(result.accepted)
        self.assertAlmostEqual(sum(result.probabilities.values()),1)
        self.assertEqual(validate_decision(result.to_dict(),observation()),result)
        data['answers']['action']['probabilities']['positive']=.85
        self.assertFalse(parse_response(data,observation()).accepted)
    def test_pooled_client_is_shared_and_requests_are_explicit(self):
        seen=[]
        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url),ALIYUN_ENDPOINT)
            self.assertEqual(request.headers['authorization'],'Bearer mock-only-placeholder')
            self.assertEqual(json.loads(request.content),request_preview(observation())['body'])
            return httpx.Response(200,json=answer())
        with patch('embodied_arm.providers.so101_jev.httpx.Client',wraps=httpx.Client) as factory:
            with self.adapter(handler=handler) as adapter:
                self.assertEqual(seen,[])
                self.assertTrue(adapter.decide(observation()).accepted)
                self.assertTrue(adapter.decide(observation()).accepted)
                self.assertEqual(factory.call_count,1)
                self.assertFalse(factory.call_args.kwargs['follow_redirects'])
                self.assertFalse(factory.call_args.kwargs['trust_env'])
            self.assertEqual(adapter.decide(observation()).reason,'closed')
        self.assertEqual(len(seen),2)
    def test_budget_busy_and_invalid_input_do_not_request(self):
        seen=[]
        def handler(request):seen.append(request);return httpx.Response(200,json=answer())
        with self.adapter(handler=handler,max_calls=1) as adapter:
            self.assertEqual(adapter.decide({}).reason,'invalid_input')
            obs=observation();obs['robot']['busy']=True
            self.assertEqual(adapter.decide(obs).reason,'busy');self.assertEqual(seen,[])
            adapter.decide(observation());self.assertEqual(adapter.decide(observation()).reason,'call_budget')
        self.assertEqual(len(seen),1)
    def test_http_timeout_and_redirect_fail_safely(self):
        def timeout(request):raise httpx.ReadTimeout('SECRET-private',request=request)
        with self.adapter(handler=timeout) as adapter:
            result=adapter.decide(observation());self.assertEqual(result.reason,'timeout');self.assertIsNone(result.action)
            self.assertNotIn('SECRET',json.dumps(result.to_dict()))
        for response,reason in [(httpx.Response(307,headers={'location':'https://evil.test'}),'api_error'),
                                (httpx.Response(401,text='SECRET'),'api_error'),
                                (httpx.Response(200,text='SECRET'),'invalid_response'),
                                (httpx.Response(200,content=b'x'*65537),'response_too_large')]:
            calls=[]
            def handler(request):calls.append(request);return response
            with self.adapter(handler=handler) as adapter:result=adapter.decide(observation())
            self.assertEqual(len(calls),1);self.assertEqual(result.reason,reason);self.assertIsNone(result.action)
    def test_explicit_config_loader_is_mocked_preserves_threshold_and_never_calls(self):
        # No real config path is opened. Only the established loader is mocked.
        with patch('embodied_arm.providers.configuration.load_configuration') as loader, patch('embodied_arm.providers.so101_jev.httpx.Client') as client:
            loader.return_value={'cerebellum':SimpleNamespace(component='cerebellum',api_key='mock-only-placeholder',endpoint=ALIYUN_ENDPOINT,
                                                         model=ALIYUN_MODEL,min_confidence=.8,timeout=15)}
            with SO101JevAdapter.from_config('/fake-explicit-path',use_environment_network=True) as adapter:
                self.assertEqual(adapter.min_confidence,.8)
                self.assertEqual(adapter.calls,0)
                self.assertEqual(adapter.max_calls,12)
                self.assertNotIn('mock-only-placeholder',json.dumps(adapter.status()))
                client.return_value.stream.assert_not_called()
                self.assertTrue(client.call_args.kwargs['trust_env'])
            loader.assert_called_once_with('/fake-explicit-path',component='cerebellum')
        with patch('embodied_arm.providers.configuration.load_configuration') as loader:
            loader.side_effect=ValueError('SECRET-config-error')
            with self.assertRaisesRegex(ValueError,'approved external') as error:
                SO101JevAdapter.from_config('/fake-explicit-path')
            self.assertNotIn('SECRET',str(error.exception))

    def test_ipc_rejection_reasons_have_no_selected_hold(self):
        for reason in ('cancelled','expired','adapter_error','worker_error'):
            decision=SO101Decision(reason=reason,axis_index=0)
            self.assertEqual(validate_decision(decision.to_dict(),observation()),decision)
        for mutate in [lambda d:d.update(axis_index=False),lambda d:d.update(confidence=False),lambda d:d.update(reason='accepted')]:
            value=SO101Decision(reason='timeout',axis_index=0).to_dict();mutate(value)
            with self.assertRaises(ValueError):validate_decision(value,observation())

    def test_configuration_and_wire_result_strictness(self):
        for kwargs in [{'min_confidence':.64},{'min_confidence':True},{'endpoint':ALIYUN_ENDPOINT+'/'},{'model':'jev-latest'}, {'timeout':31}]:
            with self.assertRaises(ValueError):self.adapter(**kwargs)
        data=SO101Decision(reason='timeout',axis_index=0).to_dict()
        self.assertIsNone(validate_decision(data,observation()).action)
        data['action']='hold'
        with self.assertRaises(ValueError):validate_decision(data,observation())
        data=parse_response(answer(),observation()).to_dict();data['action']='y_plus'
        with self.assertRaises(ValueError):validate_decision(data,observation())


if __name__=='__main__':unittest.main()
