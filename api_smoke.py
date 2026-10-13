"""Normal application CLI: one saved-credential API check, no robot simulation."""
import argparse,json,threading
from runner import JevRunner

class NoRobot:
    lock=threading.RLock()
    def stop(self):pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--environment-network',action='store_true',help='Use this execution environment\'s managed HTTPS networking; TLS verification stays enabled')
    args=parser.parse_args()
    try:
        runner=JevRunner(NoRobot(),use_environment_network=args.environment_network)
        runner.configure(args.config)
        result=runner.smoke_test()
    except Exception as e:
        result={'success':False,'status':'application_error','error_type':type(e).__name__}
    print(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
