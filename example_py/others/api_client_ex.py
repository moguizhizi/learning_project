import requests
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    request_url = f"http://localhost:8000/v1/completions"
    prompt = 'where is BeiJing'
    
    headers = {"User-Agent":"client"}
    pload = {"prompt":prompt, "n":1, "stream":False, "max_tokens":16, "temperature":0.0}
    
    response = requests.post(url=request_url, headers=headers, json=pload)
    print(response)
    data = json.loads(response.content)
    print(data)