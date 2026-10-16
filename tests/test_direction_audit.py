"""Synthetic-coordinate and mock-transport tests only; no live API or motion."""
import json
import unittest
import httpx
from adapters import ACTIONS, ALIYUN_ENDPOINT, ALIYUN_MODEL, JevAdapter
from experiments.direction_audit import batch_request, coordinate_cases, expected_direction, run_audit


def payload(wrong=None,confidence=.99):
    answers={}
    for label,case in coordinate_cases().items():
        choice=wrong if wrong else expected_direction(case)
        answers[label]={'type':'choice','choice':choice,'confidence':confidence,
                        'probabilities':{action:1 if action==choice else 0 for action in ACTIONS}}
    return {'answers':answers,'latency_ms':25.5}


class DirectionAuditTests(unittest.TestCase):
    def adapter(self,handler):
        return JevAdapter(api_key='test-only-placeholder',endpoint=ALIYUN_ENDPOINT,model=ALIYUN_MODEL,transport=httpx.MockTransport(handler))
    def test_offline_coordinate_mapping(self):
        self.assertEqual([expected_direction(case) for case in coordinate_cases().values()],
                         ['x_plus','x_minus','y_plus','y_minus','z_plus','z_minus','hold'])
        for label,case in coordinate_cases().items():
            self.assertEqual(case['current_xyz_m'],[.45,0,.3])
            self.assertAlmostEqual(sum(abs(a-b) for a,b in zip(case['current_xyz_m'],case['target_xyz_m'])),0 if label=='G' else .06)
        with self.assertRaises(ValueError):expected_direction({'current_xyz_m':[0,0,0],'target_xyz_m':[1,1,0]})
    def test_one_request_seven_answers_and_no_answer_leak(self):
        calls=[]
        def handler(request):
            calls.append(request)
            self.assertEqual(str(request.url),ALIYUN_ENDPOINT)
            body=json.loads(request.content)
            self.assertEqual(body['model'],ALIYUN_MODEL)
            self.assertEqual(set(body['questions']),set('ABCDEFG'))
            self.assertEqual(body['state'],{'cases':coordinate_cases(),'step_m':.02,'tolerance_m':.01})
            self.assertNotIn('expected',json.dumps(body))
            for label,question in body['questions'].items():
                self.assertEqual(set(question['criteria']),set(ACTIONS))
                self.assertIn('case '+label,question['instructions'])
            request.extensions['trace']('http11.send_request_body.complete',{'headers':'SECRET'})
            return httpx.Response(200,json=payload())
        result=run_audit(self.adapter(handler))
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'complete')
        self.assertEqual(result['correct'],7);self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['accepted'],7)
        self.assertEqual(result['total'],7)
        self.assertEqual(result['cases'][-1]['actual'],'hold')
        self.assertEqual(result['cases'][-1]['expected'],'hold')
        self.assertEqual(result['transport']['server_latency_ms'],25.5)
        self.assertNotIn('SECRET',json.dumps(result));self.assertNotIn('test-only-placeholder',json.dumps(result))
    def test_wrong_answers_and_low_confidence_remain_audit_only(self):
        result=run_audit(self.adapter(lambda r:httpx.Response(200,json=payload('x_plus',.35))))
        self.assertEqual(result['correct'],1);self.assertEqual(result['accepted'],0)
        self.assertEqual(result['cases'][1]['actual'],'x_plus')
        self.assertEqual(result['cases'][1]['expected'],'x_minus')
        self.assertEqual(result['cases'][1]['reason'],'low_confidence')
    def test_invalid_or_incomplete_batch_is_rejected(self):
        cases=[]
        missing=payload();del missing['answers']['F'];cases.append(missing)
        invalid=payload();invalid['answers']['A']['choice']='arbitrary';cases.append(invalid)
        invalid=payload();invalid['answers']['A']['confidence']=float('nan');cases.append(invalid)
        invalid=payload();invalid['answers']['A']['probabilities']={'x_plus':1};cases.append(invalid)
        for data in cases:
            result=run_audit(self.adapter(lambda r,d=data:httpx.Response(200,content=json.dumps(d))))
            self.assertEqual(result['status'],'invalid_response');self.assertEqual(result['cases'],[])
    def test_transport_errors_and_redirects_are_safe(self):
        def timeout(request):raise httpx.ReadTimeout('SECRET',request=request)
        result=run_audit(self.adapter(timeout))
        self.assertEqual(result['status'],'timeout');self.assertEqual(result['transport']['error_type'],'ReadTimeout')
        self.assertNotIn('SECRET',json.dumps(result))
        for response,status in [(httpx.Response(307,headers={'Location':'https://evil.test'},text='SECRET'),'http_error'),
                                (httpx.Response(200,text='SECRET'),'invalid_response'),
                                (httpx.Response(200,content=b'x'*65537),'response_too_large')]:
            calls=[]
            def handler(request):calls.append(request);return response
            result=run_audit(self.adapter(handler))
            self.assertEqual(len(calls),1);self.assertEqual(result['status'],status)
            self.assertNotIn('SECRET',json.dumps(result))


if __name__=='__main__':unittest.main()
