import streamlit as st
import requests

st.set_page_config(
    page_title="Mini DeepSeek",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Mini DeepSeek")
st.caption("Your personal AI assistant · powered by local DeepSeek via Ollama")

MODEL = "deepseek-r1:1.5b"
API_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "You are Mini DeepSeek, a helpful, concise, and friendly AI assistant. "
    "Answer clearly and helpfully. If you are unsure, say so."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Show only user & assistant messages in the UI
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask Mini DeepSeek anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            response = requests.post(
                API_URL,
                json={
                    "model": MODEL,
                    "messages": st.session_state.messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"]
            placeholder.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except requests.exceptions.ConnectionError:
            placeholder.error(
                "Could not connect to Ollama.\n\n"
                "1. Make sure Ollama is running\n"
                "2. Pull the model: `ollama pull deepseek-r1:1.5b`\n"
                "3. Then refresh this page."
            )
        except requests.RequestException as error:
            placeholder.error(f"Request failed: {error}")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Model:** `{MODEL}`")
    st.write("**Endpoint:** `localhost:11434`")
    if st.button("Clear conversation"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.rerun()
    st.markdown("---")
    st.markdown(
        "💡 Tip: You can also train the tiny educational transformer with:\n"
        "```bash\npython mini_deepseek.py\n```"
    )
