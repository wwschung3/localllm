import streamlit as st
import ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import os

# Define the configuration file paths
CONFIG_FILE = "config.json"
CONVERSATIONS_FILE = "conversations.json"
MODEL_NAME = "gpt-oss:20b"
USE_STREAM = True

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

# --- Data Persistence Functions ---
def load_config():
    """Loads application configuration from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            # Ensure all keys exist with default values if not present
            config.setdefault("system_prompt", "You are a helpful AI assistant.")
            config.setdefault("selected_language", "en")
            config.setdefault("reasoning_effort", "medium")
            config.setdefault("show_cot", False)
            config.setdefault("dark_mode", True)
            config.setdefault("history_length", 5)
            config.setdefault("auto_save", True)
            config.setdefault("profiles", {
                "Default": "You are a helpful AI assistant.",
                "Technical Assistant": "You are a highly detailed and technical assistant, specializing in programming and systems administration. Provide code examples and clear explanations.",
                "Creative Writer": "You are a creative writer who excels at crafting compelling stories, poems, and scripts. Use vivid imagery and imaginative language."
            })
            return config
    return {
        "system_prompt": "You are a helpful AI assistant.",
        "selected_language": "en",
        "reasoning_effort": "medium",
        "show_cot": False,
        "dark_mode": True,
        "history_length": 5,
        "auto_save": True,
        "profiles": {
            "Default": "You are a helpful AI assistant.",
            "Technical Assistant": "You are a highly detailed and technical assistant, specializing in programming and systems administration. Provide code examples and clear explanations.",
            "Creative Writer": "You are a creative writer who excels at crafting compelling stories, poems, and scripts. Use vivid imagery and imaginative language."
        }
    }

def save_config():
    """Saves current application configuration to a JSON file."""
    config = {
        "system_prompt": st.session_state.system_prompt,
        "selected_language": st.session_state.selected_language,
        "reasoning_effort": st.session_state.reasoning_effort,
        "show_cot": st.session_state.show_cot,
        "dark_mode": st.session_state.dark_mode,
        "auto_save": st.session_state.auto_save,
        "history_length": st.session_state.history_length,
        "profiles": st.session_state.profiles,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    st.toast("Configuration saved!")

def load_conversations():
    """Loads conversation history from a separate JSON file."""
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as f:
            serialized_conversations = json.load(f)
            deserialized_conversations = {}
            for title, history in serialized_conversations.items():
                deserialized_history = []
                for msg in history:
                    if msg['type'] == 'human':
                        deserialized_history.append(HumanMessage(content=msg['content']))
                    elif msg['type'] == 'ai':
                        deserialized_history.append(AIMessage(content=msg['content']))
                deserialized_conversations[title] = deserialized_history
            return deserialized_conversations
    return {}

def save_conversations():
    """Saves conversation history to a separate JSON file."""
    serialized_conversations = {}
    for title, history in st.session_state.conversation_titles.items():
        serialized_history = [
            {'type': 'human' if isinstance(msg, HumanMessage) else 'ai', 'content': msg.content}
            for msg in history
        ]
        serialized_conversations[title] = serialized_history
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(serialized_conversations, f, indent=4)
    st.toast("Conversations saved!")

# --- Session State Initialization ---
config = load_config()
conversations = load_conversations()

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
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None

if st.session_state.dark_mode:
    load_css('style/dark_mode.css')
else:
    load_css('style/light_mode.css')

# --- Utility Functions ---
def get_ollama_response(messages, extra_body=None):
    """Retrieves a response from the Ollama GPT-OSS model."""
    try:
        chat_model = ChatOllama(model=MODEL_NAME, extra_body={})
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"Error communicating with Ollama: {e}")
        st.warning(
            "Please ensure the 'gpt-oss' model is running locally. "
            "You can start it with: `ollama run gpt-oss`"
        )
        return "An error occurred while generating the response."

def get_ollama_stream(messages, extra_body=None):
    """
    Streams a response from an Ollama chat model, yielding only the text content.
    """
    if extra_body is None:
        extra_body = {}

    try:
        chat_model = ChatOllama(model=MODEL_NAME, **extra_body)
        stream = chat_model.stream(messages)
        for chunk in stream:
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    except Exception as e:
        st.error(f"Error streaming from Ollama: {e}")
        yield "An error occurred while streaming the response."

def generate_conversation_title(history):
    """Generates a concise title for a conversation based on its first few turns."""
    if not history:
        return "New Conversation"

    first_messages = " ".join(
        [m.content for m in history[:3] if isinstance(m, HumanMessage)]
    )
    if not first_messages:
        return "New Conversation"

    try:
        title_model = ChatOllama(model="gpt-oss")
        title_prompt = (
            f"Summarize the following conversation snippet into a very concise title "
            f"(under 8 words, without quotes or conversational phrases): "
            f"'{first_messages}'"
        )
        title_response = title_model.invoke(
            [
                SystemMessage(content="You are a helpful summarizer."),
                HumanMessage(content=title_prompt),
            ]
        )
        return (
            title_response.content.strip()
            .replace('"', "")
            .replace(".", "")
            .replace(":", "")
            .split("\n")[0]
        )
    except Exception as e:
        print(f"Error generating title: {e}")
        return f"Conversation {len(st.session_state.conversation_titles) + 1}"

def save_current_conversation():
    """Saves the current conversation. If one is loaded, it overwrites it. Otherwise, a new one is created."""
    if not st.session_state.chat_history:
        st.warning("Cannot save an empty conversation.")
        return

    if st.session_state.current_conversation_title:
        title_to_save = st.session_state.current_conversation_title
        st.session_state.conversation_titles[title_to_save] = list(st.session_state.chat_history)
        st.toast(f"Conversation '{title_to_save}' updated!")
    else:
        title = generate_conversation_title(st.session_state.chat_history)
        st.session_state.conversation_titles[title] = list(st.session_state.chat_history)
        st.session_state.current_conversation_title = title
        st.toast(f"Conversation '{title}' saved!")
    
    save_conversations()
    st.rerun()

def load_conversation(title):
    """Loads a previously saved conversation into the current chat."""
    if title in st.session_state.conversation_titles:
        st.session_state.chat_history = list(
            st.session_state.conversation_titles[title]
        )
        st.session_state.current_conversation_title = title
        st.toast(f"Conversation '{title}' loaded!")
        st.rerun()

def rename_conversation_handler(old_title, new_title):
    """Renames a saved conversation."""
    if old_title != new_title and new_title not in st.session_state.conversation_titles:
        st.session_state.conversation_titles[new_title] = st.session_state.conversation_titles.pop(old_title)
        if st.session_state.current_conversation_title == old_title:
            st.session_state.current_conversation_title = new_title
        save_conversations()
        st.session_state.rename_mode = False
        st.toast(f"Conversation '{old_title}' renamed to '{new_title}'!")
        st.rerun()
    elif old_title == new_title:
        st.toast("Name is the same. No changes made.")
    else:
        st.warning(f"A conversation with the name '{new_title}' already exists.")

def delete_conversation(title):
    """Deletes a saved conversation."""
    if title in st.session_state.conversation_titles:
        del st.session_state.conversation_titles[title]
        if st.session_state.current_conversation_title == title:
            st.session_state.chat_history = []
            st.session_state.current_conversation_title = None
        save_conversations()
        st.toast(f"Conversation '{title}' deleted!")
        st.rerun()

# --- Consolidated Sidebar ---
with st.sidebar:
    st.header("Configuration")
    st.markdown("---")

    st.subheader("File Upload")
    uploaded_file = st.file_uploader(
        "Upload a text file (text file only):",
        key="file_uploader"
    )
    if uploaded_file:
        try:
            file_content = uploaded_file.read().decode("utf-8")
            st.session_state.uploaded_file_content = file_content
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("---")

    st.subheader("System Prompt Profiles")
    if st.button("New Profile", use_container_width=True, key="new_profile_btn"):
        st.session_state.selected_profile_name = "New Profile"
        st.session_state.system_prompt = "You are a helpful AI assistant."
        st.rerun()

    if st.session_state.profiles:
        profile_names = list(st.session_state.profiles.keys())
        for profile_name in profile_names:
            if st.button(profile_name, key=f"profile_button_{profile_name}", use_container_width=True):
                st.session_state.selected_profile_name = profile_name
                st.session_state.system_prompt = st.session_state.profiles[profile_name]
                st.rerun()
    st.markdown("---")

    st.subheader("Edit Current Profile")
    profile_name_input = st.text_input(
        "Profile Name:",
        st.session_state.selected_profile_name,
        key="profile_name_input"
    )
    current_system_prompt = st.text_area(
        "Define the AI's persona or instructions:",
        st.session_state.system_prompt,
        height=150,
        key="system_prompt_input"
    )
    col_actions = st.columns(2)
    with col_actions[0]:
        if st.button("Save Profile", use_container_width=True, key="save_profile_btn"):
            if not profile_name_input:
                st.warning("Please enter a profile name.")
            else:
                st.session_state.profiles[profile_name_input] = current_system_prompt
                st.session_state.selected_profile_name = profile_name_input
                st.session_state.system_prompt = current_system_prompt
                save_config()
                st.rerun()
    with col_actions[1]:
        if st.button("Delete Profile", use_container_width=True, key="delete_profile_btn"):
            if st.session_state.selected_profile_name in st.session_state.profiles:
                if st.session_state.selected_profile_name == "Default":
                    st.warning("Cannot delete the 'Default' profile.")
                else:
                    del st.session_state.profiles[st.session_state.selected_profile_name]
                    st.session_state.selected_profile_name = "Default"
                    st.session_state.system_prompt = st.session_state.profiles.get("Default", "You are a helpful AI assistant.")
                    save_config()
                    st.rerun()

    st.markdown("---")
    
    auto_save_checked = st.sidebar.checkbox(
        "Auto‑Save conversations", value=st.session_state.auto_save, key="auto_save_checkbox"
    )
    if auto_save_checked != st.session_state.auto_save:
        st.session_state.auto_save = auto_save_checked
        save_config()

    st.subheader("Previous Conversations")
    col_save, col_new = st.columns(2)
    with col_save:
        if st.button("Save Current", key="save_conv_btn", use_container_width=True):
            save_current_conversation()
    with col_new:
        if st.button("New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_conversation_title = None
            st.session_state.uploaded_file_content = None
            st.rerun()

    if st.session_state.rename_mode:
        with st.form("rename_form", clear_on_submit=False):
            st.write(f"Rename conversation: **{st.session_state.conversation_to_rename}**")
            new_name = st.text_input("New name:", st.session_state.conversation_to_rename, key="rename_input_modal")
            col_modal_1, col_modal_2 = st.columns(2)
            with col_modal_1:
                if st.form_submit_button("Save", use_container_width=True):
                    rename_conversation_handler(st.session_state.conversation_to_rename, new_name)
            with col_modal_2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.rename_mode = False
                    st.session_state.conversation_to_rename = ""
                    st.rerun()

    if st.session_state.conversation_titles:
        for title in st.session_state.conversation_titles:
            is_active = (title == st.session_state.current_conversation_title)
            with st.container():
                st.markdown(f"**{'🟢 ' if is_active else ''}{title}**")
                col_load, col_rename, col_delete = st.columns(3)
                with col_load:
                    if st.button("Load", key=f"load_conv_{title}", use_container_width=True):
                        load_conversation(title)
                with col_rename:
                    if st.button("Rename", key=f"rename_btn_{title}", use_container_width=True):
                        st.session_state.rename_mode = True
                        st.session_state.conversation_to_rename = title
                        st.rerun()
                with col_delete:
                    if st.button("Delete", key=f"delete_btn_{title}", use_container_width=True):
                        delete_conversation(title)
                st.markdown("---")
                
        if st.button("Clear All Saved Conversations", key="clear_all_convs", use_container_width=True):
            st.session_state.conversation_titles = {}
            st.session_state.chat_history = []
            st.session_state.current_conversation_title = None
            st.session_state.uploaded_file_content = None
            save_conversations()
            st.toast("All conversations cleared!")
            st.rerun()
    else:
        st.info("No conversations saved yet.")
    
    st.markdown("---")
    st.header("Settings")
    st.markdown("---")

    st.subheader("Display Theme")
    theme_choice = st.selectbox(
        "Choose a theme:",
        ("Dark", "Light"),
        index=0 if st.session_state.dark_mode else 1,
        key="theme_select",
    )
    new_dark_mode = theme_choice == "Dark"
    if new_dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = new_dark_mode
        save_config()
        st.rerun()

    st.markdown("---")

    st.subheader("Response Language")
    selected_language_from_ui = st.selectbox(
        "Choose the language for AI responses:",
        ("en", "zh-tw"),
        index=("en", "zh-tw").index(st.session_state.selected_language),
        key="language_select",
    )
    if selected_language_from_ui != st.session_state.selected_language:
        st.session_state.selected_language = selected_language_from_ui
        save_config()

    st.markdown("---")

    st.subheader("Conversation History")
    history_length_from_ui = st.slider(
        "Number of previous messages to remember:",
        min_value=1,
        max_value=20,
        value=st.session_state.history_length,
        step=1,
        key="history_length_slider"
    )
    if history_length_from_ui != st.session_state.history_length:
        st.session_state.history_length = history_length_from_ui
        save_config()

    st.markdown("---")

    st.subheader("Chain-of-Thought")
    show_cot_from_ui = st.checkbox(
        "Show chain‑of‑thought", value=st.session_state.show_cot, key="cot_checkbox"
    )
    if show_cot_from_ui != st.session_state.show_cot:
        st.session_state.show_cot = show_cot_from_ui
        save_config()

    st.markdown("---")

    st.subheader("Reasoning Effort")
    reasoning_effort_from_ui = st.selectbox(
        "Set the reasoning level:",
        ("low", "medium", "high"),
        index=("low", "medium", "high").index(st.session_state.reasoning_effort),
        key="reasoning_effort_select",
    )
    if reasoning_effort_from_ui != st.session_state.reasoning_effort:
        st.session_state.reasoning_effort = reasoning_effort_from_ui
        save_config()

# --- Main Content Area ---
st.title("Private AI Playground")
st.caption("Model: " + MODEL_NAME)

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

user_input = st.chat_input("You:")
if user_input:
    # Combine user input and uploaded file content, if any
    combined_input = user_input
    if "uploaded_file_content" in st.session_state and st.session_state.uploaded_file_content:
        combined_input += "\n\n[Uploaded File Content]:\n" + st.session_state.uploaded_file_content
        # Clear the uploaded file content after use
        st.session_state.uploaded_file_content = None

    st.chat_message("user").write(combined_input)
    st.session_state.chat_history.append(HumanMessage(content=combined_input))

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            system_prompt = st.session_state.system_prompt
            language_instruction = "Respond in English."
            if st.session_state.selected_language == "zh-tw":
                language_instruction = "Respond in Traditional Chinese."
            system_prompt += f" {language_instruction}"

            if st.session_state.show_cot:
                if st.session_state.selected_language == "zh-tw":
                    system_prompt += "\n\n請先以 '思考過程：' 開頭解釋你的推理和思考流程，然後再以 '最終答案：' 開頭給出最終答案。"
                else:
                    system_prompt += "\n\nFirst, explain your reasoning and thought process starting with 'Thought:'. Then, provide your final answer starting with 'Answer:'."

            extra_body = {
                "reasoning_effort": st.session_state.reasoning_effort
            }
            
            prompt = [SystemMessage(content=system_prompt)]
            prompt.extend(st.session_state.chat_history[-st.session_state.history_length:])

            if not USE_STREAM:
                response = get_ollama_response(prompt, extra_body=extra_body)
                st.write(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            else:
                response_placeholder = st.empty()
                full_text = ""
                for token in get_ollama_stream(prompt, extra_body=extra_body):
                    full_text += token
                    response_placeholder.write(full_text)
                st.session_state.chat_history.append(AIMessage(content=full_text))
            
            if st.session_state.auto_save:
                save_current_conversation()