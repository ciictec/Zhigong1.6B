from transformers import AutoTokenizer, LlamaForCausalLM
import re
from typing import Tuple, List, Dict
import torch
import streamlit as st
from streamlit_chat import message
from loguru import logger
import math

model_name = "Assistant"
human_prompt = "<|Human|>"
assistant_prompt = f"<|{model_name}|>"

model_path = "ciictec/Zhigong-1.6B-Chat"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Zhigong-1.6B-Chat",
    page_icon=":robot:"
) 
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
    def chat(self, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, top_k=50, temperature=0.8, repetition_penalty=1.0, **kwargs):
        if history is None:
            history = []
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
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

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, add_bos_token=False)
    model = ModelChat.from_pretrained(model_path, tokenizer=tokenizer).half().to(device)
    
    return tokenizer, model

def predict(query, max_length, top_p, top_k, temperature, repetition_penalty, num_beams, history=None):
    tokenizer, model = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            for i, round_history in enumerate(history):
                history_query, history_response = round_history
                message(history_query.replace(human_prompt, ""), avatar_style="big-smile", key=str(i) + "_user")
                message(history_response.replace(assistant_prompt, "").replace(tokenizer.eos_token, "").replace(tokenizer.bos_token, "").strip(), avatar_style="bottts", key=str(i))

        message(query, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            response, history = model.chat(query=query, history=history, max_length=max_length,
                                           top_p=top_p, top_k=top_k, temperature=temperature,
                                           repetition_penalty=repetition_penalty, num_beams=num_beams)
            st.write(response)
    return history


        
container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")

max_length = st.sidebar.slider(
    'max_length', 0, 2048, 2048, step=1
)
top_p = st.sidebar.slider(
    'top_p', 0.0, 1.0, 0.7, step=0.01
)
top_k = st.sidebar.slider(
    'top_k', 0, 100, 30, step=1
)
temperature = st.sidebar.slider(
    'temperature', 0.0, 30.0, 0.8, step=0.01
)
repetition_penalty = st.sidebar.slider(
    'repetition_penalty', 1.0, 1.10, 1.0, step=0.01
)
num_beams = st.sidebar.slider(
    'num_beams', 1, 10, 1, step=1
)

if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        st.session_state["state"] = predict(prompt_text, max_length, top_p, top_k, temperature,
                                            repetition_penalty, num_beams, st.session_state["state"])
