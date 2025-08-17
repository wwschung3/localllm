# ollama_client.py
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

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


def get_ollama_response(model_name: str, messages, extra_body: dict | None = None):
    """
    取得 Ollama GPT‑OSS 模型的回覆。
    """
    try:
        chat_model = _build_chat_model(model_name, extra_body)
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        st.warning(
            "Please ensure the 'gpt-oss' model is running locally. "
            "You can start it with: `ollama run gpt-oss`"
        )
        return "An error occurred while generating the response."


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