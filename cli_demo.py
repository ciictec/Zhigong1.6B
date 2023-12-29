import re
from typing import Tuple, List
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

model_name = "Assistant"
human_prompt = "<|Human|>"
assistant_prompt = f"<|{model_name}|>"

class ModelChat(LlamaForCausalLM):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens": [human_prompt, assistant_prompt]})
        self.history = []

    def process_response(self, response):
        response = response.strip()
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        response = response.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.bos_token, "").strip()
        return response

    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, max_new_tokens: int = 1024, num_beams=1,
             do_sample=True, top_p=0.7, top_k=30, temperature=0.8, repetition_penalty=1.0, **kwargs):
        if history is None:
            history = []
        gen_kwargs = {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "top_k": top_k, "temperature": temperature, "repetition_penalty": repetition_penalty,
                      "bos_token_id": self.tokenizer.bos_token_id, "eos_token_id": self.tokenizer.eos_token_id, **kwargs}
        history_prompt = ""
        for turn in history:
            history_prompt += turn[0] + turn[1]
        wrap_query = human_prompt + str(query)
        prompt = history_prompt + wrap_query + assistant_prompt 
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)
        response = self.process_response(response)
        history.append([wrap_query, assistant_prompt + response + self.tokenizer.eos_token])
        
        return response, history

if __name__ == '__main__':
    model_path = "ciictec/Zhigong-1.6B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, add_bos_token=False)
    model = ModelChat.from_pretrained(model_path, tokenizer=tokenizer, device_map="auto").half()

    while True:
        q = input("User: ")
        response, history = model.chat(query=q, max_new_tokens=512,history=None)
        print("Zhigong-1.6B-Chat: ")
        print(response)
        print('\n')

