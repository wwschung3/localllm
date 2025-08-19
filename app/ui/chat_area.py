import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import persistence, ollama_client
import config as default_config
import pyperclip

def _get_unique_id() -> int:
	"""
	Return a unique integer that can be used as a widget key.
	We simply use the current length of the chat history.
	"""
	return len(st.session_state.chat_history)


def _copy_button(text: str, idx: int) -> None:
	"""
	Renders a Copy button that copies the given text to the clipboard.
	
	Parameters
	----------
	text : str
		The content of the chat message.
	idx : int
		Position of the message in the chat history (used for the button key).
	"""
	button_key = f"copy_{idx}"
	
	# Use a unique session state key for each button's content
	session_state_key = f"text_to_copy_{button_key}"

	if st.button(label="📋 Copy", key=button_key):
		# When the button is clicked, store the text with a unique key.
		st.session_state[session_state_key] = text
		st.success(f"Text copied to clipboard!")
	
	# Handle the actual copy action via JavaScript
	if session_state_key in st.session_state:
		try:
			# Try the built-in clipboard support first.
			pyperclip.copy(st.session_state[session_state_key])
			# Clean up the unique session state key after copying.
			del st.session_state[session_state_key]
		except Exception as e:
			st.toast(f"Error when copying text: {e}")
			del st.session_state[session_state_key]

# --------------------------------------------------------------------------- #
#  Main: 渲染整個聊天介面
# --------------------------------------------------------------------------- #
def render_chatarea() -> None:
	st.title("Private AI Playground")
	st.caption(f"Model: {default_config.MODEL_NAME}")

	# 迭代聊天歷史，並在每條訊息下方放置「複製」按鈕
	for idx, msg in enumerate(st.session_state.chat_history):
		if isinstance(msg, HumanMessage):
			with st.chat_message("user"):
				st.write(msg.content)
				_copy_button(msg.content, idx)
		elif isinstance(msg, AIMessage):
			with st.chat_message("assistant"):
				st.write(msg.content)
				_copy_button(msg.content, idx)
		else:
			# 其他可能的訊息類型（若有）
			with st.chat_message("assistant"):
				st.write(json.dumps(msg.__dict__, indent=2))
				_copy_button(json.dumps(msg.__dict__, indent=2), idx)

	user_input = st.chat_input("You:")
	if user_input:
		# Combine user input and uploaded file content, if any
		combined_input = user_input
		# Check if uploaded_file_data is a list of tuples and not empty
		if "uploaded_file_data" in st.session_state and st.session_state.uploaded_file_data:
			formatted_file_contents = []
			for file_name, file_content in st.session_state.uploaded_file_data:
				formatted_file_contents.append(f"--- File: {file_name} ---\n{file_content}")
			
			all_file_contents = "\n\n".join(formatted_file_contents)
			combined_input += "\n\n[Uploaded File Contents]:\n" + all_file_contents

		st.chat_message("user").write(combined_input)
		st.session_state.chat_history.append(HumanMessage(content=combined_input))
		new_idx = _get_unique_id() - 1  # the index of the just‑added message
		_copy_button(combined_input, new_idx)

		# Clear the uploaded file data after use
		st.session_state.uploaded_file_data = []
		st.session_state.file_uploader_id = st.session_state.file_uploader_id + 1

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

				PARAMS_BY_EFFORT = {
					"low":	{"temperature": 0.2,  "top_p": 0.95, "frequency_penalty": 0.7, "presence_penalty": 0.7,  "max_tokens": 350},
					"medium": {"temperature": 0.5,  "top_p": 0.90, "frequency_penalty": 0.4, "presence_penalty": 0.4,  "max_tokens": 2000},
					"high":   {"temperature": 0.8,  "top_p": 0.80, "frequency_penalty": 0.2, "presence_penalty": 0.2,  "max_tokens": 20000},
				}
				params = PARAMS_BY_EFFORT[st.session_state.reasoning_effort]
				extra_body = {"reasoning_effort": st.session_state.reasoning_effort, **params}
				system_prompt += f"\nMust use this reasoning_effort: {st.session_state.reasoning_effort};"

				prompt = [SystemMessage(content=system_prompt)]
				prompt.extend(st.session_state.chat_history[-st.session_state.history_length:])

				if not default_config.USE_STREAM:
					response = ollama_client.get_ollama_response(default_config.MODEL_NAME, prompt, extra_body=extra_body)
					st.write(response)
					st.session_state.chat_history.append(AIMessage(content=response))
				else:
					response_placeholder = st.empty()
					full_text = ""
					for token in ollama_client.get_ollama_stream(default_config.MODEL_NAME, prompt, extra_body=extra_body):
						full_text += token
						response_placeholder.write(full_text)
					st.session_state.chat_history.append(AIMessage(content=full_text))
				
				if st.session_state.auto_save:
					persistence.save_current_conversation()