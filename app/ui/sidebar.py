import streamlit as st
from utils import persistence
import tiktoken
import json
import streamlit.components.v1 as components
from rag.embedding_model import embedding_model
from rag.vector_store_manager import vector_store_manager

def process_uploaded_files(uploaded_files):
	"""Processes uploaded text files, calculates token counts, and updates session state.
	If token count exceeds 2000, it processes them with RAG."""
	if not uploaded_files:
		st.session_state.uploaded_file_data = []
		st.session_state.file_token_counts = {}
		vector_store_manager.clear_index() # Clear RAG index if no files are uploaded
		return

	st.session_state.uploaded_file_data = []
	st.session_state.file_token_counts = {}
	
	# Initialize RAG components
	embedding_model.load()
	vector_store_manager.init_vector_store(dim=embedding_model.model.get_sentence_embedding_dimension()) # Use model's dimension

	doc_id_counter = 0 # Unique ID for each chunk

	for uploaded_file in uploaded_files:
		try:
			file_content = None
			encodings_to_try = ['utf-8', 'big5', 'gbk', 'gb2312', 'latin-1']

			for encoding in encodings_to_try:
				try:
					# Move the file pointer back to the beginning before each decode attempt
					uploaded_file.seek(0)
					file_content = uploaded_file.read().decode(encoding)
					break  # Exit the loop if decoding is successful
				except UnicodeDecodeError:
					continue  # Try the next encoding

			if file_content is not None:
				# Use file_content here
				st.success("File decoded successfully!")
			else:
				st.error("Could not decode the file. The encoding may be unsupported.")
			tokens = st.session_state.token_encoder.encode(file_content)
			token_count = len(tokens)

			st.session_state.uploaded_file_data.append((uploaded_file.name, file_content))
			st.session_state.file_token_counts[uploaded_file.name] = token_count

			st.success(f"File '{uploaded_file.name}' uploaded successfully! Tokens: **{token_count}**")

			total_token_count = sum(st.session_state.file_token_counts.values())

			# If token count exceeds 1500, process with RAG
			if total_token_count > 1500:
				st.info(f"File '{uploaded_file.name}' is large ({token_count} tokens). Processing for RAG...")
				# For simplicity, we'll treat the entire file as one chunk for now
				# In a real-world scenario, you'd chunk this file into smaller pieces
				# and add each chunk to the vector store.
				
				# Embed the content
				vector = embedding_model.embed_text(file_content)
				
				# Add to vector store
				vector_store_manager.add_document(doc_id_counter, vector, file_content)
				doc_id_counter += 1 # Increment for the next chunk/file
				vector_store_manager.save_metadata() # Save metadata after adding document
				st.success(f"File '{uploaded_file.name}' processed for RAG.")

		except Exception as e:
			st.error(f"Error reading file '{uploaded_file.name}': {e}")

	
def render_sidebar():
	st.header("Configuration")
	st.markdown("---")

	st.subheader("File Upload")

	# Initialize token encoder in session state if it doesn't exist
	if "token_encoder" not in st.session_state:
		st.session_state.token_encoder = tiktoken.get_encoding("cl100k_base")

	# The file uploader widget
	uploaded_files = st.file_uploader(
		"Upload text files (content will be used as context for the next query):",
		accept_multiple_files=True,
		key=f"file_uploader_{st.session_state.file_uploader_id}"
	)

	# Process files when the uploader state changes
	process_uploaded_files(uploaded_files)

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
		current_profile = st.session_state.get("selected_profile_name", "")

		js_value = json.dumps(current_profile)

		# Render the script with a zero‑height component so it doesn't affect layout.
		components.html(
			f"""
			<script>
			// When the iframe finishes loading, expose the value to the parent page
			document.addEventListener("DOMContentLoaded", function() {{
				// Set the variable on the parent (the main page)
				window.parent.selectedProfileName = {js_value};
				console.log("parent.selectedProfileName set to:", window.parent.selectedProfileName);
			}});
			</script>
			""",
			height=0,  # invisible
		)
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
				persistence.save_config()
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
					persistence.save_config()
					st.rerun()

	st.markdown("---")
	
	auto_save_checked = st.sidebar.checkbox(
		"Auto‑Save conversations", value=st.session_state.auto_save, key="auto_save_checkbox"
	)
	if auto_save_checked != st.session_state.auto_save:
		st.session_state.auto_save = auto_save_checked
		persistence.save_config()

	st.subheader("Previous Conversations")
	col_save, col_new = st.columns(2)
	with col_save:
		if st.button("Save Current", key="save_conv_btn", use_container_width=True):
			persistence.save_current_conversation()
	with col_new:
		if st.button("New Chat", key="new_chat_btn", use_container_width=True):
			st.session_state.chat_history = []
			st.session_state.current_conversation_title = None
			st.session_state.uploaded_file_data = []
			vector_store_manager.clear_index() # Clear RAG index on new chat
			st.rerun()

	if st.session_state.rename_mode:
		with st.form("rename_form", clear_on_submit=False):
			st.write(f"Rename conversation: **{st.session_state.conversation_to_rename}**")
			new_name = st.text_input("New name:", st.session_state.conversation_to_rename, key="rename_input_modal")
			col_modal_1, col_modal_2 = st.columns(2)
			with col_modal_1:
				if st.form_submit_button("Save", use_container_width=True):
					persistence.rename_conversation_handler(st.session_state.conversation_to_rename, new_name)
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
						persistence.load_conversation(title)
				with col_rename:
					if st.button("Rename", key=f"rename_btn_{title}", use_container_width=True):
						st.session_state.rename_mode = True
						st.session_state.conversation_to_rename = title
						st.rerun()
				with col_delete:
					if st.button("Delete", key=f"delete_btn_{title}", use_container_width=True):
						persistence.delete_conversation(title)
				st.markdown("---")
				
		if st.button("Clear All Saved Conversations", key="clear_all_convs", use_container_width=True):
			st.session_state.conversation_titles = {}
			st.session_state.chat_history = []
			st.session_state.current_conversation_title = None
			st.session_state.uploaded_file_data = []
			vector_store_manager.clear_index() # Clear RAG index on clearing all conversations
			persistence.save_conversations()
			st.toast("All conversations cleared!")
			st.rerun()
	else:
		st.info("No conversations saved yet.")
	
	st.markdown("---")
	st.header("Settings")
	st.markdown("---")

	st.subheader("Search Internet")
	use_search_from_ui = st.checkbox(
		"Enable Search Internet", value=st.session_state.use_search, key="use_search_checkbox"
	)
	if use_search_from_ui != st.session_state.use_search:
		st.session_state.use_search = use_search_from_ui
		persistence.save_config()

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
		persistence.save_config()
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
		persistence.save_config()

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
		persistence.save_config()

	st.markdown("---")

	st.subheader("Chain-of-Thought")
	show_cot_from_ui = st.checkbox(
		"Show chain‑of‑thought", value=st.session_state.show_cot, key="cot_checkbox"
	)
	if show_cot_from_ui != st.session_state.show_cot:
		st.session_state.show_cot = show_cot_from_ui
		persistence.save_config()

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
		persistence.save_config()
