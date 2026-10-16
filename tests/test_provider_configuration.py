import json
import httpx
import pytest
from embodied_arm.providers.configuration import BASE_URL,parse_configuration,load_configuration,normalize_base_url
from embodied_arm.providers.brain import QwenBrain
from embodied_arm.providers.vision_adapter import QwenVision
from embodied_arm.providers.so101_jev import SO101JevAdapter
from embodied_arm.providers.diagnostics import check_connection,main


def config():
 return {'schema_version':2,'brain':{'provider':'aliyun_bailian','base_url':BASE_URL,'model':'qwen3.8-max','api_key':'fake-brain-key','timeout':24},
         'cerebellum':{'provider':'aliyun_bailian','base_url':BASE_URL,'model':'decision-model-preview','api_key':'fake-jev-key','timeout':9,'min_confidence':.7}}

def test_independent_settings_and_vision_defaults_to_brain(tmp_path):
 path=tmp_path/'providers.json';path.write_text(json.dumps(config()));before=path.read_bytes()
 brain=QwenBrain.from_config(path);vision=QwenVision.from_config(path)
 with SO101JevAdapter.from_config(path) as jev:
  assert brain._api_key==vision._api_key=='fake-brain-key' and jev._key=='fake-jev-key'
  assert brain.timeout==vision.timeout==24 and jev.min_confidence==.7
 assert path.read_bytes()==before
 public=''.join(str(x.public_status())+repr(x) for x in load_configuration(path).values())
 assert 'fake-brain-key' not in public and 'fake-jev-key' not in public

def test_legacy_maps_in_memory_without_rewrite(tmp_path):
 legacy={'provider':'aliyun_bailian','endpoint':BASE_URL+'/systemone','model':'decision-model-preview','api_key':'fake-shared-key'}
 path=tmp_path/'legacy.json';path.write_text(json.dumps(legacy));before=path.read_bytes()
 groups=load_configuration(path)
 assert groups['brain'].api_key==groups['cerebellum'].api_key=='fake-shared-key'
 assert groups['brain'].endpoint.endswith('/chat/completions') and groups['cerebellum'].endpoint.endswith('/systemone')
 assert before==path.read_bytes()

@pytest.mark.parametrize('component,route',[('brain','/chat/completions'),('cerebellum','/systemone')])
@pytest.mark.parametrize('form',['base','slash','route','route_slash'])
def test_endpoint_normalization_never_double_appends(component,route,form):
 value={'base':BASE_URL,'slash':BASE_URL+'/','route':BASE_URL+route,'route_slash':BASE_URL+route+'/'}[form]
 assert normalize_base_url(value,component)==BASE_URL

@pytest.mark.parametrize('url',['http://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1',BASE_URL+'?key=secret',BASE_URL+'#x',BASE_URL+'/systemone','https://evil.invalid/compatible-mode/v1','https://user:secret@token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'])
def test_wrong_destination_or_protocol_rejected(url):
 data=config();data['brain']['base_url']=url
 with pytest.raises(ValueError):parse_configuration(data)

@pytest.mark.parametrize('edit',[lambda d:d.update(schema_version=True),lambda d:d.update(extra='x'),lambda d:d['brain'].update(model='other'),lambda d:d['brain'].update(api_key='secret\nkey'),lambda d:d['cerebellum'].update(min_confidence=.5),lambda d:d['brain'].update(timeout=float('inf'))])
def test_invalid_groups_rejected(edit):
 data=config();edit(data)
 with pytest.raises(ValueError):parse_configuration(data)

def test_duplicate_fields_and_in_repo_paths_rejected_without_secret_echo(tmp_path):
 path=tmp_path/'duplicate.json';path.write_text('{"api_key":"secret","api_key":"other"}')
 with pytest.raises(ValueError) as e:load_configuration(path)
 assert 'secret' not in str(e.value)
 with pytest.raises(ValueError):load_configuration('config/provider-v2.example.json')

@pytest.mark.parametrize('component',['brain','cerebellum'])
def test_each_connection_probe_uses_own_key_endpoint_model_and_one_call(component):
 settings=parse_configuration(config())[component];calls=[]
 def respond(request):
  calls.append(request);assert str(request.url)==settings.endpoint
  assert request.headers['authorization']=='Bearer '+settings.api_key
  assert json.loads(request.content)['model']==settings.model
  payload={'model':settings.model,'choices':[{'finish_reason':'stop','message':{'content':'OK'}}]} if component=='brain' else {'answers':{'connection':{'choice':'red'}}}
  return httpx.Response(200,json=payload)
 result=check_connection(settings,transport=httpx.MockTransport(respond),use_environment_network=False)
 assert result['ok'] and result['calls']==len(calls)==1
 assert settings.api_key not in str(result)

@pytest.mark.parametrize('status',[301,401,403,429,500])
def test_probe_does_not_follow_redirect_retry_or_echo_response(status):
 calls=[];settings=parse_configuration(config())['brain']
 def respond(request):calls.append(request);return httpx.Response(status,headers={'Location':'https://evil.invalid'},json={'error':settings.api_key})
 result=check_connection(settings,transport=httpx.MockTransport(respond))
 assert result['reason']==f'http_error_{status}' and len(calls)==1 and settings.api_key not in str(result)

def test_cli_defaults_to_local_validation_without_network(tmp_path,monkeypatch,capsys):
 path=tmp_path/'providers.json';path.write_text(json.dumps(config()))
 monkeypatch.setattr(httpx,'Client',lambda **k:pytest.fail('No implicit requests'))
 assert main(['--config',str(path),'--component','brain'])==0
 result=json.loads(capsys.readouterr().out)
 assert result['calls']==0 and result['operation']=='configuration_validation'

def test_one_component_can_be_used_when_other_key_is_not_configured(tmp_path):
 data=config();data['cerebellum']['api_key']=''
 path=tmp_path/'partial.json';path.write_text(json.dumps(data))
 assert QwenBrain.from_config(path)._api_key=='fake-brain-key'
 assert QwenVision.from_config(path)._api_key=='fake-brain-key'
 with pytest.raises(ValueError):SO101JevAdapter.from_config(path)

@pytest.mark.parametrize('cap',[True,0,float('nan'),float('inf')])
def test_invalid_runtime_timeout_cap_does_not_silently_disappear(tmp_path,cap):
 path=tmp_path/'providers.json';path.write_text(json.dumps(config()))
 with pytest.raises(ValueError):QwenBrain.from_config(path,timeout=cap)
 with pytest.raises(ValueError):SO101JevAdapter.from_config(path,timeout=cap)
