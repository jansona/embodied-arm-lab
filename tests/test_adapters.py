"""Offline protocol and safety tests. No actual API calls or credentials."""
import json
import unittest

import httpx

from embodied_arm.providers.adapters import ACTIONS, ALIYUN_ENDPOINT, ALIYUN_MODEL, Decision, JevAdapter, OFFICIAL_ENDPOINT, OFFICIAL_MODEL, request_preview


def answer(choice="x_plus", confidence=0.9):
    probabilities = {a: 0.0125 for a in ACTIONS}
    probabilities[choice] = 0.9
    return {"answers": {"action": {"type": "choice", "choice": choice,
            "confidence": confidence, "probabilities": probabilities}}}


class AdapterTests(unittest.TestCase):
    def adapter(self, payload=None, handler=None, **kwargs):
        if handler is None:
            handler = lambda request: httpx.Response(200, json=payload)
        return JevAdapter(api_key="test-only-placeholder", transport=httpx.MockTransport(handler), **kwargs)

    def test_preview_is_detached_and_unauthenticated(self):
        state = {"position": [0, 0, 1], "goal": "pick and place"}
        preview = request_preview(state)
        self.assertEqual(preview["endpoint"], OFFICIAL_ENDPOINT)
        self.assertEqual(preview["body"]["model"], "jev-latest")
        self.assertEqual(set(preview["body"]["questions"]["action"]["criteria"]), set(ACTIONS))
        preview["body"]["state"]["position"][0] = 99
        self.assertEqual(state["position"][0], 0)
        self.assertNotIn("Authorization", json.dumps(preview))

    def test_no_network_until_explicit_decide_and_correct_request(self):
        seen = []
        def handler(request):
            seen.append(request)
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), OFFICIAL_ENDPOINT)
            self.assertEqual(request.headers["authorization"], "Bearer test-only-placeholder")
            self.assertEqual(json.loads(request.content), request_preview({"step": 1})["body"])
            self.assertEqual(request.extensions["timeout"]["read"], 8.0)
            return httpx.Response(200, json=answer())
        adapter = self.adapter(handler=handler)
        self.assertEqual(seen, [])
        result = adapter.decide({"step": 1})
        self.assertTrue(result.accepted)
        self.assertEqual(result.action, "x_plus")
        self.assertEqual(len(seen), 1)
        self.assertNotIn("test-only-placeholder", repr(adapter))

    def test_explicit_aliyun_pair_request_body_and_auth(self):
        seen = []
        state = {'step': 1, 'simulation_only': True}
        preview = request_preview(state, endpoint=ALIYUN_ENDPOINT, model=ALIYUN_MODEL)
        self.assertEqual(preview['endpoint'], ALIYUN_ENDPOINT)
        self.assertEqual(preview['body']['model'], ALIYUN_MODEL)
        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url), ALIYUN_ENDPOINT)
            self.assertEqual(request.headers['authorization'], 'Bearer test-only-placeholder')
            self.assertEqual(json.loads(request.content), preview['body'])
            return httpx.Response(200, json=answer())
        adapter = self.adapter(handler=handler, endpoint=ALIYUN_ENDPOINT, model=ALIYUN_MODEL)
        self.assertEqual(seen, [])
        self.assertTrue(adapter.decide(state).accepted)
        self.assertEqual(len(seen), 1)

    def test_provider_pair_and_lookalike_urls_rejected(self):
        pairs = [(ALIYUN_ENDPOINT, OFFICIAL_MODEL), (OFFICIAL_ENDPOINT, ALIYUN_MODEL),
                 (ALIYUN_ENDPOINT + '/', ALIYUN_MODEL),
                 (ALIYUN_ENDPOINT + '?redirect=evil', ALIYUN_MODEL),
                 (ALIYUN_ENDPOINT.replace('https:', 'http:'), ALIYUN_MODEL),
                 (ALIYUN_ENDPOINT.replace('.aliyuncs.com', '.aliyuncs.com.evil.test'), ALIYUN_MODEL),
                 (ALIYUN_ENDPOINT.replace('token-plan.', 'another-workspace.'), ALIYUN_MODEL),
                 (ALIYUN_ENDPOINT.replace('https://', 'https://user@'), ALIYUN_MODEL),
                 (ALIYUN_ENDPOINT, 'unexpected-model')]
        for endpoint, model in pairs:
            with self.subTest(endpoint=endpoint, model=model):
                with self.assertRaises(ValueError):
                    self.adapter(endpoint=endpoint, model=model)
                with self.assertRaises(ValueError):
                    request_preview({}, endpoint=endpoint, model=model)

    def test_api_only_smoke_test_exact_red_blue_body(self):
        seen = []
        def handler(request):
            seen.append(request)
            self.assertEqual(str(request.url), ALIYUN_ENDPOINT)
            self.assertEqual(request.headers['authorization'], 'Bearer test-only-placeholder')
            self.assertEqual(json.loads(request.content), {
                'model': ALIYUN_MODEL, 'state': {'color': 'red'}, 'questions': {
                    'color': {'type': 'choice', 'instructions': 'What color is specified in state?',
                              'criteria': {'red': 'The color is red', 'blue': 'The color is blue'}}}})
            return httpx.Response(200, json={'answers': {'color': {
                'type': 'choice', 'choice': 'red', 'confidence': .99,
                'probabilities': {'red': 1, 'blue': 0}}}})
        result = self.adapter(handler=handler, endpoint=ALIYUN_ENDPOINT, model=ALIYUN_MODEL).smoke_test()
        self.assertEqual(len(seen), 1)
        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['choice'], 'red')
        self.assertEqual(result['http_status'], 200)
        self.assertGreaterEqual(result['latency_seconds'], 0)
        self.assertNotIn('test-only-placeholder', json.dumps(result))

    def test_smoke_test_failures_are_sanitized(self):
        def network_failure(request):
            try:
                raise OSError(111, 'SECRET-token and private URL')
            except OSError as cause:
                raise httpx.ConnectError('SECRET-token', request=request) from cause
        result = self.adapter(handler=network_failure).smoke_test()
        self.assertFalse(result['success'])
        self.assertEqual(result['status'], 'network_error')
        self.assertEqual(result['error_type'], 'ConnectError')
        self.assertEqual(result['cause_type'], 'ConnectionRefusedError')
        self.assertEqual(result['errno'], 111)
        self.assertNotIn('SECRET', json.dumps(result))
        for response, status in [(httpx.Response(401, text='SECRET-body'), 'http_error'),
                                 (httpx.Response(307, headers={'Location': 'https://evil.test'}), 'http_error'),
                                 (httpx.Response(200, text='SECRET-nonjson'), 'invalid_response'),
                                 (httpx.Response(200, content=b'x' * 65537), 'response_too_large')]:
            calls = []
            def handler(request): calls.append(request); return response
            result = self.adapter(handler=handler).smoke_test()
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['status'], status)
            self.assertFalse(result['success'])
            self.assertNotIn('SECRET', json.dumps(result))

    def test_smoke_test_validates_answer_and_wrong_color(self):
        valid = {'type': 'choice', 'choice': 'red', 'confidence': .9,
                 'probabilities': {'red': 1, 'blue': 0}}
        malformed = [None, {}, {**valid, 'choice': 'green'}, {**valid, 'confidence': float('nan')},
                     {**valid, 'probabilities': {'red': .3, 'blue': .7}},
                     {**valid, 'probabilities': {'red': 1}}, {**valid, 'type': 'score'}]
        for data in malformed:
            result = self.adapter(handler=lambda r, d=data: httpx.Response(200, content=json.dumps({'answers': {'color': d}}))).smoke_test()
            self.assertEqual(result['status'], 'invalid_response')
            self.assertFalse(result['success'])
        blue = {**valid, 'choice': 'blue', 'probabilities': {'red': 0, 'blue': 1}}
        result = self.adapter({'answers': {'color': blue}}).smoke_test()
        self.assertEqual(result['status'], 'unexpected_choice')
        self.assertFalse(result['success'])

    def test_hover_prompt_is_explicit_and_keeps_all_actions(self):
        preview = request_preview({'objective': {'id': 'hover_reach', 'target_xyz': [.5,0,.44], 'tolerance_m': .01}, 'end_effector': [.412345,-.01,.43], 'workspace': [[.25,.7],[-.35,.35],[.022,.6]], 'goal': [0,0,0], 'joints': [0]*7, 'metrics': {'actions':3}})
        compact=preview['body']['state']
        self.assertEqual(set(compact),{'task','current_xyz_m','target_xyz_m','delta_xyz_m','step_m','tolerance_m','workspace'})
        self.assertEqual(compact['current_xyz_m'],[.4123,-.01,.43])
        self.assertEqual(compact['target_xyz_m'],[.5,0,.44])
        self.assertEqual(compact['delta_xyz_m'],[.0877,.01,.01])
        self.assertNotIn('x_plus',json.dumps(compact))
        self.assertNotIn('goal',compact)
        self.assertNotIn('joints',compact)
        question = preview['body']['questions']['action']
        self.assertEqual(set(question['criteria']), set(ACTIONS))
        self.assertIn('largest absolute coordinate error', question['instructions'])
        self.assertIn('Do not open or close', question['instructions'])
        self.assertEqual(self.adapter(answer()).min_confidence, .65)
        self.assertEqual(self.adapter(answer()).decide({}).proposed_action, 'x_plus')

    def test_trace_events_are_name_and_time_only(self):
        class SecretInfo:
            def __repr__(self): raise AssertionError('Trace info must never be inspected')
        for smoke in (False,True):
            def handler(request):
                trace=request.extensions['trace']
                for name in ['connection.connect_tcp.started','connection.connect_tcp.complete',
                             'http11.send_request_body.complete','http11.receive_response_headers.started']:
                    trace(name,SecretInfo())
                trace('http11.SECRET-token.complete',{'Authorization':'SECRET-token'})
                for _ in range(40):trace('http11.receive_response_body.started',SecretInfo())
                payload=answer() if not smoke else {'answers':{'color':{'type':'choice','choice':'red','confidence':.9,'probabilities':{'red':1,'blue':0}}}}
                payload['latency_ms']=23.5
                return httpx.Response(200,json=payload)
            adapter=self.adapter(handler=handler)
            result=adapter.smoke_test() if smoke else adapter.decide({})
            diagnostics=adapter.last_transport
            self.assertEqual(diagnostics['server_latency_ms'],23.5)
            self.assertEqual(len(diagnostics['events']),30)
            for key in ['request_start','request_body_sent','response_headers_received','response_complete','elapsed_seconds']:
                self.assertGreaterEqual(diagnostics[key],0)
            self.assertNotIn('SECRET',json.dumps(diagnostics))
            if smoke:self.assertEqual(result['transport'],diagnostics)
            diagnostics['events'].clear()
            self.assertEqual(len(adapter.last_transport['events']),30)

    def test_trace_read_timeout_shows_stage_without_exception_text(self):
        def handler(request):
            trace=request.extensions['trace']
            trace('http11.send_request_body.complete',{'request':'SECRET'})
            trace('http11.receive_response_headers.started',{'request':'SECRET'})
            raise httpx.ReadTimeout('SECRET body and endpoint',request=request)
        adapter=self.adapter(handler=handler)
        self.assertEqual(adapter.decide({}).reason,'timeout')
        self.assertEqual(adapter.last_transport['error_type'],'ReadTimeout')
        self.assertIn('request_body_sent',adapter.last_transport)
        self.assertNotIn('response_headers_received',adapter.last_transport)
        self.assertNotIn('SECRET',json.dumps(adapter.last_transport))

    def test_low_confidence_holds(self):
        decision = self.adapter(answer(confidence=0.2)).decide({})
        self.assertEqual(decision.action, "hold")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "low_confidence")
        self.assertEqual(decision.confidence, 0.2)
        self.assertEqual(decision.proposed_action, "x_plus")
        self.assertEqual(decision.probabilities["x_plus"], .9)

    def test_invalid_answers_hold(self):
        invalid = [None, [], {}, {"answers": []}, {"answers": {"action": []}}]
        for key, value in [("type", "score"), ("choice", "move_100m"), ("choice", []),
                           ("confidence", -1), ("confidence", 10**1000), ("confidence", 2), ("confidence", True),
                           ("confidence", "0.9"), ("confidence", float("nan")),
                           ("confidence", float("inf")), ("probabilities", None),
                           ("probabilities", {"x_plus": 1})]:
            data = answer()
            data["answers"]["action"][key] = value
            invalid.append(data)
        for mutation in [lambda p: p.update(x_plus=0.1), lambda p: p.update(x_minus=-0.1),
                         lambda p: p.update(x_minus=True), lambda p: p.update(x_minus=float("nan")),
                         lambda p: p.update(extra=0)]:
            data = answer()
            mutation(data["answers"]["action"]["probabilities"])
            invalid.append(data)
        wrong_max = answer()
        wrong_max["answers"]["action"]["choice"] = "hold"
        invalid.append(wrong_max)
        for data in invalid:
            with self.subTest(data=data):
                # json.dumps intentionally permits NaN to exercise malformed remote JSON.
                result = self.adapter(handler=lambda r, d=data: httpx.Response(200, content=json.dumps(d))).decide({})
                self.assertEqual(result, Decision(reason="invalid_response"))

    def test_every_allowed_action(self):
        for action in ACTIONS:
            with self.subTest(action=action):
                self.assertEqual(self.adapter(answer(action)).decide({}).action, action)

    def test_bad_state_does_not_send(self):
        def forbidden(request):
            self.fail("Invalid state should never reach the transport")
        for state in [[], None, {"x": float("nan")}, {"x": object()}]:
            with self.subTest(state=state):
                self.assertEqual(self.adapter(handler=forbidden).decide(state).reason, "invalid_state")

    def test_timeout_and_network_errors_are_sanitized(self):
        for error, reason in [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "network_error")]:
            def handler(request, exc=error):
                raise exc("secret-key and private-response-body", request=request)
            result = self.adapter(handler=handler).decide({})
            self.assertEqual(result.reason, reason)
            self.assertNotIn("secret", json.dumps(result.to_dict()))

    def test_redirect_is_not_followed(self):
        seen = []
        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(307, headers={"location": "https://example.com/steal"})
        result = self.adapter(handler=handler).decide({})
        self.assertEqual(result.reason, "http_error_307")
        self.assertEqual(seen, [OFFICIAL_ENDPOINT])

    def test_http_and_invalid_json_and_size_errors(self):
        cases = [(httpx.Response(401, text="private-response-body"), "http_error_401"),
                 (httpx.Response(200, text="not json"), "invalid_response"),
                 (httpx.Response(200, content=b"x" * 65537), "response_too_large")]
        for response, reason in cases:
            with self.subTest(reason=reason):
                result = self.adapter(handler=lambda r, response=response: response).decide({})
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.action, "hold")
                self.assertNotIn("private-response-body", str(result))

    def test_configuration_is_explicit_and_restricted(self):
        for endpoint in ["http://api.typesafe.ai/v1/systemone", "https://evil.test", OFFICIAL_ENDPOINT + "?key=x", OFFICIAL_ENDPOINT + "/"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                JevAdapter(api_key="test-only", endpoint=endpoint)
        for key in [None, "", "  ", "token\nInjected: x", "ë"]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                JevAdapter(api_key=key)
        for confidence in [-1, 2, True, "0.5", float("nan")]:
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                JevAdapter(api_key="test-only", min_confidence=confidence)
        for timeout in [0, -1, 61, True, "8", float("inf")]:
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                JevAdapter(api_key="test-only", timeout=timeout)


if __name__ == "__main__":
    unittest.main()

class RoundedDistributionTests(unittest.TestCase):
    def test_provider_two_decimal_rounding_is_bounded(self):
        adapter=JevAdapter(api_key='offline-test-only')
        p={a:0 for a in ACTIONS};p['z_minus']=.91;p['hold']=.05;p['z_plus']=.03
        answer={'answers':{'action':{'type':'choice','choice':'z_minus','confidence':.9,'probabilities':p}}}
        decision=adapter._validate(answer)
        self.assertTrue(decision.accepted)
        self.assertAlmostEqual(sum(decision.probabilities.values()),1)
        self.assertEqual(decision.confidence,.9)
        p['z_minus']=.70
        self.assertFalse(adapter._validate(answer).accepted)
        p['z_minus']=.90001
        self.assertFalse(adapter._validate(answer).accepted)
