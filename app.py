
import streamlit as st
import requests

st.set_page_config(
    page_title="Mini DeepSeek",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Mini DeepSeek")
st.caption("Your personal AI assistant running locally")

MODEL = "deepseek-r1:1.5b"
API_URL = "http://localhost:11434/api/chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask Mini DeepSeek anything...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            response = requests.post(
                API_URL,
                json={
                    "model": MODEL,
                    "messages": st.session_state.messages,
                    "stream": False
                },
                timeout=300
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"]
            placeholder.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except requests.RequestException as error:
            placeholder.error(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            )
            st.caption(str(error))
