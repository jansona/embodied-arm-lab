"""Credential-free diagnostic; never reads secret configuration."""
import json,os,socket,ssl
import httpx
HOST='token-plan.cn-beijing.maas.aliyuncs.com'
URL='https://'+HOST+'/compatible-mode/v1/systemone'
def check():
    out={'host':HOST,'authenticated':False,'request_method':'GET','proxy_environment_present':any(k in os.environ for k in ('HTTPS_PROXY','https_proxy','ALL_PROXY','all_proxy'))}
    try:socket.getaddrinfo(HOST,443);out['dns']='resolved'
    except OSError as e:out['dns']=type(e).__name__
    try:
        with httpx.Client(timeout=5,follow_redirects=False,trust_env=False) as c:
            r=c.get(URL);out['https_status']=r.status_code
    except Exception as e:
        out['https_error_type']=type(e).__name__
        causes=[];p=e
        for _ in range(4):
            if p is None:break
            causes.append(type(p).__name__);p=p.__cause__
        out['cause_types']=causes
    try:
        context=ssl.create_default_context()
        with socket.create_connection((HOST,443),timeout=5) as sock:
            with context.wrap_socket(sock,server_hostname=HOST) as secure:
                out['system_tls']='verified'
    except Exception as e:
        out['system_tls_error']=type(e).__name__
        out['system_tls_errno']=getattr(e,'errno',None)
        out['verify_code']=getattr(e,'verify_code',None)
        out['verify_message']=getattr(e,'verify_message',None)
    return out
if __name__=='__main__':print(json.dumps(check()))
