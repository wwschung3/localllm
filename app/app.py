import streamlit as st
import ollama
from langchain_community.chat_models import ChatOllama
import json
import os
from ui import sidebar, chat_area
import config as default_config
from utils import persistence

# --- Page Configuration and CSS Injection ---
st.set_page_config(
    page_title="Local LLM Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load and inject CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- Session State Initialization ---
config = persistence.load_config()
conversations = persistence.load_conversations()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = config["system_prompt"]
if "selected_language" not in st.session_state:
    st.session_state.selected_language = config["selected_language"]
if "reasoning_effort" not in st.session_state:
    st.session_state.reasoning_effort = config["reasoning_effort"]
if "show_cot" not in st.session_state:
    st.session_state.show_cot = config["show_cot"]
if "conversation_titles" not in st.session_state:
    st.session_state.conversation_titles = conversations
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = config["dark_mode"]
if "history_length" not in st.session_state:
    st.session_state.history_length = config["history_length"]
if "profiles" not in st.session_state:
    st.session_state.profiles = config["profiles"]
if "selected_profile_name" not in st.session_state:
    st.session_state.selected_profile_name = "Default"
if "rename_mode" not in st.session_state:
    st.session_state.rename_mode = False
if "conversation_to_rename" not in st.session_state:
    st.session_state.conversation_to_rename = ""
if "current_conversation_title" not in st.session_state:
    st.session_state.current_conversation_title = None
if "auto_save" not in st.session_state:
    st.session_state.auto_save = config["auto_save"]
if "use_search" not in st.session_state:
    st.session_state.use_search = config["use_search"]
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = [] # Initialize as a list

if st.session_state.dark_mode:
    load_css('style/dark_mode.css')
else:
    load_css('style/light_mode.css')


# --- Consolidated Sidebar ---
with st.sidebar:
    sidebar.render_sidebar()

# --- Main Content Area ---
chat_area.render_chatarea()
