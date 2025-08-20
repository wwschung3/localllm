# ollama_client.py
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ---------- 版本兼容處理 ----------
# 新版 (>=0.3.1) 使用 langchain_ollama
# 舊版則使用 langchain_community.chat_models
try:
    # 新版匯入
    from langchain_ollama import ChatOllama
except ImportError:
    # 舊版備援
    from langchain_community.chat_models import ChatOllama

# ---------- 工具函式 ----------
def _build_chat_model(model_name: str, extra_body: dict | None):
    """
    建立 ChatOllama 實例，並且把 user 直接傳入的 extra_body
    作為關鍵字參數傳給建構子。若新版不支援該參數，
    則會被忽略 (因為 **kwargs 會忽略不存在的參數)。
    """
    kwargs = extra_body or {}
    # 新版建構子通常使用 `model_kwargs`，舊版使用 `extra_body`
    # 為了簡化，直接把所有 key 當成關鍵字參數傳遞即可
    return ChatOllama(model=model_name, **kwargs)

def convert_messages_to_string_simple(messages):
    return "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in messages])

def get_ollama_response(model_name: str, user_input: str, extra_body: dict | None = None):
    """
    Connects to an Ollama instance, sends a single string prompt (derived from messages),
    and yields the response content using the chat.completions.create method.

    Args:
        model_name (str): The name of the Ollama model to use (e.g., "llama2", "gpt-oss").
        messages (list): A list of message objects (e.g., from Langchain),
                         each having 'type' (e.g., 'human', 'ai', 'system')
                         and 'content' attributes. These will be converted to a single string.
        extra_body (dict | None): Optional dictionary for additional parameters
                                  to be passed to the API request body.

    Yields:
        str: Chunks of the generated response content.
    """
    try:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  # Ollama doesn't require a real API key; this is a placeholder.
        )

        # Call the chat completions API
        stream = client.completions.create(
            model=model_name,
            prompt=user_input,
            max_tokens=16384,
            temperature=0.6,
            top_p=0.95,
            stream=True,
            extra_body=extra_body or {}
        )

        for chunk in stream:
            # Access the 'text' attribute instead of 'delta'.
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text
    except Exception as e:
        # Using st.error and st.warning assumes a Streamlit environment
        st.error(f"Error communicating with Ollama: {e}")
        st.warning(
            "Please ensure the specified Ollama model is running locally. "
            "You can start it with: `ollama run {model_name}`"
        )
        # Re-raise the exception after logging/displaying in Streamlit
        yield f"Exception: {e}"



def get_ollama_stream(model_name: str, messages, extra_body: dict | None = None):
    """
    逐塊串流回覆。回傳的內容只包含純文字。
    """
    if extra_body is None:
        extra_body = {}

    try:
        chat_model = _build_chat_model(model_name, extra_body)
        stream = chat_model.stream(messages)
        for chunk in stream:
            # 兼容多種 chunk 物件結構
            content = getattr(chunk, "content", None)
            if content:
                yield content
    except Exception as e:
        st.error(f"Error streaming from Ollama: {e}")
        yield "An error occurred while streaming the response."