import streamlit as st
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import persistence, ollama_client, prompt_builder
import config as default_config
import pyperclip
from rag.embedding_model import embedding_model # Import embedding_model
from rag.vector_store_manager import vector_store_manager # Import vector_store_manager

top_k = 20

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

	# Initialize last_uploaded_filename if not present
	if 'last_uploaded_filename' not in st.session_state:
		st.session_state.last_uploaded_filename = None

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
		# This is the message that will be displayed to the user
		display_input = user_input
		# This is the context that will be passed to the LLM via the system prompt
		context_for_llm = None 
		
		# If RAG is enabled, perform a search based on the user's query
		if st.session_state.rag_enabled:
			st.info("Searching RAG context...")
			# Embed the user's query
			query_vector = embedding_model.embed_text(user_input)
			# Search the vector store for relevant chunks
			# The search results now include filename: (doc_id, score, filename)
			results = vector_store_manager.search(query_vector, k=10) # Retrieve more chunks to allow for prioritization
			
			if results:
				# Prioritization Logic:
				# If user mentions "this file", "last file", or the specific last uploaded filename,
				# prioritize chunks from that file.
				prioritized_results = []
				other_results = []
				
				# Keywords to detect intent for the last uploaded file
				user_input_lower = user_input.lower()
				referring_to_last_file = False
				if st.session_state.last_uploaded_filename:
					if "this file" in user_input_lower or \
					   "the last file" in user_input_lower or \
					   st.session_state.last_uploaded_filename.lower() in user_input_lower:
						referring_to_last_file = True

				if referring_to_last_file:
					st.info(f"Prioritizing content from '{st.session_state.last_uploaded_filename}' based on your query.")
					for doc_id, score, filename in results:
						if filename == st.session_state.last_uploaded_filename:
							prioritized_results.append((doc_id, score, filename))
						else:
							other_results.append((doc_id, score, filename))
					# Combine prioritized results first, then other results, up to k
					# Ensure unique doc_ids if there's overlap (though unlikely with distinct lists)
					final_results = prioritized_results + other_results
				else:
					final_results = results # No specific prioritization, use original search results

				# Limit to top K chunks for the LLM context
				final_results = final_results[:top_k]

				retrieved_texts = []
				for doc_id, score, filename in final_results: # Iterate through prioritized results
					# Retrieve the actual text content from metadata
					metadata_entry = vector_store_manager.metadata.get(doc_id)
					if metadata_entry and 'text' in metadata_entry:
						text_content = metadata_entry['text']
						retrieved_texts.append(f"--- File: {filename} (Score: {score:.4f}) ---\n{text_content}")
				
				if retrieved_texts:
					formatted_rag_context = "\n\n---\n\n".join(retrieved_texts)
					display_input += "\n\n[Relevant Context from Uploaded Files]:\n" + formatted_rag_context
					context_for_llm = formatted_rag_context
					st.session_state.rag_context = retrieved_texts # Store for potential future display/debug if needed
					st.success("RAG context found and added to prompt.")
				else:
					st.warning("No relevant RAG context found for your query after prioritization.")
					context_for_llm = None # Ensure it's None if no relevant context
			else:
				st.warning("No RAG index available or no results found. Using raw uploaded content if available.")
				context_for_llm = None # Ensure it's None if no results

		# If RAG was NOT enabled OR no RAG context was found, check for raw uploaded file data
		# and pass it as context. This acts as a fallback or for smaller files.
		if not context_for_llm and st.session_state.uploaded_file_data:
			formatted_file_contents = []
			for file_name, file_content in st.session_state.uploaded_file_data:
				formatted_file_contents.append(f"--- File: {file_name} ---\n{file_content}")
			
			all_file_contents = "\n\n".join(formatted_file_contents)
			display_input += "\n\n[Uploaded File Contents]:\n" + all_file_contents
			context_for_llm = all_file_contents # This will be the context for the LLM
			st.session_state.rag_context = [] # Clear RAG context if raw files are used


		# Display the combined input to the user
		st.chat_message("user").write(display_input)
		new_idx = _get_unique_id() # Get unique ID before appending
		_copy_button(display_input, new_idx)

		# Append ONLY the original user_input to chat_history for LLM's conversational memory
		st.session_state.chat_history.append(HumanMessage(content=user_input))

		# Clear the uploaded file data after use, regardless of RAG
		st.session_state.uploaded_file_data = []
		st.session_state.file_uploader_id = st.session_state.file_uploader_id + 1 # Increment to clear uploader visually


		with st.chat_message("assistant"):
			with st.spinner("思考中…"):
				# Construct prompt using the prompt_builder module
				# Pass the separate context_for_llm to prompt_builder
				prompt = prompt_builder.build_prompt(
					st.session_state.system_prompt,
					st.session_state.selected_language,
					st.session_state.show_cot,
					st.session_state.reasoning_effort,
					st.session_state.chat_history, # This now contains ONLY clean chat turns
					rag_context=context_for_llm # This is the retrieved RAG or raw file content
				)

				PARAMS_BY_EFFORT = {
					"low":	{"temperature": 0.2,  "top_p": 0.95, "frequency_penalty": 0.7, "presence_penalty": 0.7,  "max_tokens": 350},
					"medium": {"temperature": 0.5,  "top_p": 0.90, "frequency_penalty": 0.4, "presence_penalty": 0.4,  "max_tokens": 2000},
					"high":   {"temperature": 0.8,  "top_p": 0.80, "frequency_penalty": 0.2, "presence_penalty": 0.2,  "max_tokens": 20000},
				}
				params = PARAMS_BY_EFFORT[st.session_state.reasoning_effort]
				extra_body = {"reasoning_effort": st.session_state.reasoning_effort, **params}
				
				if not default_config.USE_STREAM:
					response = ollama_client.get_ollama_response(default_config.MODEL_NAME, user_input, extra_body=extra_body)
					st.write(response)
					#st.session_state.chat_history.append(AIMessage(content=response))
				else:
					response_placeholder = st.empty()
					full_text = ""
					for token in ollama_client.get_ollama_stream(default_config.MODEL_NAME, prompt, extra_body=extra_body):
						full_text += token
						response_placeholder.write(full_text)
					st.session_state.chat_history.append(AIMessage(content=full_text))
				
				if st.session_state.auto_save:
					persistence.save_current_conversation()
