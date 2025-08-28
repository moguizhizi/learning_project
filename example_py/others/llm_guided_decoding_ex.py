from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import GuidedDecodingParams

def main():
    llm = LLM(model="/data/llm_model/huggingface/TinyLlama--TinyLlama-1.1B-Chat-v1.0/", guided_decoding_backend="xgrammar")
    
    schema = '{"title": "WirelessAccessPoint", "type": "object", "properties": {"ssid": {"title": "SSID", "type": "string"}, "securityProtocol": {"title": "SecurityProtocol", "type": "string"}, "bandwidth": {"title": "Bandwidth", "type": "string"}}, "required": ["ssid", "securityProtocol", "bandwidth"]}'

    prompt = [{
        'role':
        'system',
        'content':
        "You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{'title': 'WirelessAccessPoint', 'type': 'object', 'properties': {'ssid': {'title': 'SSID', 'type': 'string'}, 'securityProtocol': {'title': 'SecurityProtocol', 'type': 'string'}, 'bandwidth': {'title': 'Bandwidth', 'type': 'string'}}, 'required': ['ssid', 'securityProtocol', 'bandwidth']}\n</schema>\n"
    }, {
        'role':
        'user',
        'content':
        "I'm currently configuring a wireless access point for our office network and I need to generate a JSON object that accurately represents its settings. The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps on the 5 GHz band. This JSON object will be used to document our network configurations and to automate the setup process for additional access points in the future. Please provide a JSON object that includes these details."
    }] 
    
    prompt = llm.tokenizer.apply_chat_template(prompt, tokenize=False)
    print(f"prompt:{prompt!r}")
    
    print("result(no guided)")
    result = llm.generate(prompt, sampling_params=SamplingParams(max_tokens=50))
    
    print(result.outputs[0].text)
    
    print("result(guided)")
    result = llm.generate(prompt, sampling_params=SamplingParams(max_tokens=50, guided_decoding=GuidedDecodingParams(json=schema)))
    
    print(result.outputs[0].text)
    
    

if __name__ == '__main__':
    main()
