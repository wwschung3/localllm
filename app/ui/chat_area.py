import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import persistence, ollama_client
import config as default_config

def render_chatarea():
	st.title("Private AI Playground")
	st.caption("Model: " + default_config.MODEL_NAME)

	for idx, msg in enumerate(st.session_state.chat_history):
		# 1️⃣ 先顯示訊息本體
		if isinstance(msg, HumanMessage):
			with st.chat_message("user"):
				st.write(msg.content)
				# 2️⃣ 在使用者訊息後面加上 Copy 按鈕
				copy_html = _copy_button_html(msg.content, idx, "user")
				st.markdown(copy_html, unsafe_allow_html=True)
		else:
			with st.chat_message("assistant"):
				st.write(msg.content)
				# 2️⃣ 在 AI 訊息後面加上 Copy 按鈕
				copy_html = _copy_button_html(msg.content, idx, "assistant")
				st.markdown(copy_html, unsafe_allow_html=True)

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
			# Clear the uploaded file data after use
			st.session_state.uploaded_file_data = []


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