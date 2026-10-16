"""Explicit per-component diagnostics. Importing or starting UI never calls this.

CLI defaults to local configuration validation. --test-connection makes exactly
one authenticated request for the selected fixed provider; it cannot move a robot.
"""
import argparse
import json
import httpx
from embodied_arm.providers.configuration import load_configuration,normalize_base_url,MODELS

def check_connection(settings,*,transport=None,use_environment_network=True):
    component=settings.component
    if component not in MODELS or settings.model!=MODELS[component]:raise ValueError('unsupported_component')
    normalize_base_url(settings.base_url,component)
    result={**settings.public_status(),'operation':'connection_test','calls':0,'ok':False,'reason':None}
    if component=='brain':
        body={'model':settings.model,'messages':[{'role':'user','content':'Connection check only. Reply OK.'}],
              'stream':False,'enable_thinking':False,'max_tokens':16}
    else:
        body={'model':settings.model,'state':{'color':'red'},'questions':{'connection':{
            'type':'choice','instructions':'Select the color named in state.color.',
            'criteria':{'red':'state.color is red','blue':'state.color is blue'}}}}
    try:
        with httpx.Client(timeout=min(settings.timeout,10),follow_redirects=False,
                          trust_env=use_environment_network,transport=transport) as client:
            result['calls']=1
            with client.stream('POST',settings.endpoint,headers={'Authorization':'Bearer '+settings.api_key,'Accept':'application/json'},json=body) as response:
                result['http_status']=response.status_code
                if response.status_code!=200:
                    result['reason']='http_error_'+str(response.status_code);return result
                chunks=[];size=0
                for chunk in response.iter_bytes():
                    size+=len(chunk)
                    if size>65536:result['reason']='response_too_large';return result
                    chunks.append(chunk)
                payload=json.loads(b''.join(chunks))
        if component=='brain':
            valid=(payload.get('model')==settings.model and isinstance(payload.get('choices'),list)
                   and bool(payload['choices']) and payload['choices'][0].get('finish_reason')=='stop'
                   and isinstance(payload['choices'][0].get('message',{}).get('content'),str))
        else:
            valid=payload.get('answers',{}).get('connection',{}).get('choice')=='red'
        result.update(ok=bool(valid),reason='connected' if valid else 'invalid_provider_response')
    except httpx.TimeoutException:result['reason']='timeout'
    except httpx.HTTPError:result['reason']='network_error'
    except (ValueError,TypeError,KeyError,AttributeError,IndexError):result['reason']='invalid_provider_response'
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--component',choices=('brain','cerebellum'),required=True)
    parser.add_argument('--test-connection',action='store_true',help='Explicit1-call provider test; may incur a charge')
    args=parser.parse_args(argv)
    try:
        settings=load_configuration(args.config,component=args.component)[args.component]
        result=check_connection(settings) if args.test_connection else {
            **settings.public_status(),'operation':'configuration_validation','ok':True,'calls':0}
    except ValueError:
        result={'component':args.component,'ok':False,'reason':'configuration_invalid','calls':0}
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['ok'] else 1

if __name__=='__main__':raise SystemExit(main())
