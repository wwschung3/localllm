import streamlit as st
import json
import os
from langchain_core.messages import HumanMessage, AIMessage
import config as default_config

# ---- config ----
def load_config(file_path: str=default_config.CONFIG_FILE):
    """Loads application configuration from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            config = json.load(f)
            # Ensure all keys exist with default values if not present
            config.setdefault("system_prompt", "You are a helpful AI assistant.")
            config.setdefault("selected_language", "en")
            config.setdefault("reasoning_effort", "medium")
            config.setdefault("show_cot", False)
            config.setdefault("theme", True)
            config.setdefault("history_length", 5)
            config.setdefault("auto_save", True)
            config.setdefault("profiles", {
                "Default": "You are a helpful AI assistant.",
                "Technical Assistant": "You are a highly detailed and technical assistant, specializing in programming and systems administration. Provide code examples and clear explanations.",
                "Creative Writer": "You are a creative writer who excels at crafting compelling stories, poems, and scripts. Use vivid imagery and imaginative language."
            })
            return config
    return {
        "system_prompt": default_config.DEFAULT_SYSTEM_PROMPT,
        "selected_language": default_config.DEFAULT_SELECTED_LANGUAGE,
        "reasoning_effort": default_config.DEFAULT_REASONING_EFFORT,
        "show_cot": default_config.DEFAULT_SHOW_COT,
        "dark_mode": default_config.DEFAULT_DARK_MODE,
        "history_length": default_config.DEFAULT_HISTORY_LENGTH,
        "auto_save": default_config.DEFAULT_AUTO_SAVE,
        "profiles": default_config.DEFAULT_PROFILES
    }

def save_config(file_path: str=default_config.CONFIG_FILE):
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
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)
    st.toast("Configuration saved!")

# ---- conversations ----

def load_conversations(file_path: str=default_config.CONVERSATIONS_FILE):
    """Loads conversation history from a separate JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
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

def save_conversations(file_path: str=default_config.CONVERSATIONS_FILE):
    """Saves conversation history to a separate JSON file."""
    serialized_conversations = {}
    for title, history in st.session_state.conversation_titles.items():
        serialized_history = [
            {'type': 'human' if isinstance(msg, HumanMessage) else 'ai', 'content': msg.content}
            for msg in history
        ]
        serialized_conversations[title] = serialized_history
    with open(file_path, "w") as f:
        json.dump(serialized_conversations, f, indent=4)
    st.toast("Conversations saved!")

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