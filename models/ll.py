import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
from vllm import LLM, SamplingParams
import os
from qhuglm import llm

# llm = LLM(model="THUDM/chatglm3-6b", trust_remote_code=True)

if 'message' not in st.session_state:
    st.session_state.message = []

for message in st.session_state.message:
    with st.chat_message(message['role']):
        st.markdown(message['text'], unsafe_allow_html=True)


def generate_text(text: str):
    prompts = [
        f"<|user|>\n{text}\n<|user|>\n",
    ]
    sampling_params = SamplingParams(temperature=0.5, top_p=0.8, max_tokens=8192)
    outputs = llm.generate(prompts, sampling_params)
    return outputs[0].outputs[0].text.strip()


# Streamlit app
st.title("QHU-GPT ChatBot", '')

# Text input for user prompt
prompt = st.text_input("Enter a prompt:", "")

if prompt:
    with st.spinner("Wait a moment..."):
        text = generate_text(prompt)
    st.chat_message('user').markdown(prompt, unsafe_allow_html=True)
    st.session_state.message.append(
        dict(
            role='user',
            text=prompt
        )
    )
    st.chat_message('bot').markdown(text, unsafe_allow_html=True)
    st.session_state.message.append(
        dict(
            role='bot',
            text=text
        )
    )
