import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "heegyu/llama-small-randomweights"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("AI Chatbot")
st.write("This is a chatbot powered by the Llama-small model from Hugging Face.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(f"You: {user_input}")
        response = generate_response(user_input)
        st.session_state.chat_history.append(f"Bot: {response}")

if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.write(chat)
