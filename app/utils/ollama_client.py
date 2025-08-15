import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

def get_ollama_response(model_name: str, messages, extra_body=None):
    """Retrieves a response from the Ollama GPT-OSS model."""
    try:
        chat_model = ChatOllama(model=model_name, extra_body={})
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        st.warning(
            "Please ensure the 'gpt-oss' model is running locally. "
            "You can start it with: `ollama run gpt-oss`"
        )
        return "An error occurred while generating the response."

def get_ollama_stream(model_name: str, messages, extra_body=None):
    """
    Streams a response from an Ollama chat model, yielding only the text content.
    """
    if extra_body is None:
        extra_body = {}

    try:
        chat_model = ChatOllama(model=model_name, **extra_body)
        stream = chat_model.stream(messages)
        for chunk in stream:
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    except Exception as e:
        st.error(f"Error streaming from Ollama: {e}")
        yield "An error occurred while streaming the response."
