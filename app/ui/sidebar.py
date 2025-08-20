import streamlit as st
from utils import persistence
import tiktoken
import json
import streamlit.components.v1 as components
from rag.embedding_model import embedding_model
from rag.vector_store_manager import vector_store_manager
# Assuming a simple text chunking strategy for demonstration
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import tqdm
import csv # Import the csv module
import io # Import io for string-based file handling


def _process_csv_file(uploaded_file, file_content_raw, all_content_for_rag_processing):
	"""
	Processes a single CSV file, extracts rows, and formats them for RAG.
	Appends formatted rows to all_content_for_rag_processing.
	"""
	st.info(f"Processing '{uploaded_file.name}' as CSV...")
	csv_file = io.StringIO(file_content_raw)
	csv_reader = csv.reader(csv_file)
	
	headers = next(csv_reader, None) # Read header row
	if headers:
		for row_idx, row in enumerate(csv_reader):
			# Create a formatted string for each row
			formatted_row = f"--- File: {uploaded_file.name} (Row {row_idx + 2}) ---\n" # +2 for header and 0-index
			row_dict = {}
			for i, header in enumerate(headers):
				if i < len(row): # Ensure row has data for this header
					row_dict[header] = row[i]
			formatted_row += json.dumps(row_dict, ensure_ascii=False, indent=2) # Use JSON for structured row
			
			all_content_for_rag_processing.append({
				'text': formatted_row,
				'filename': uploaded_file.name,
				'original_content': formatted_row # Store the formatted row as the content to be embedded
			})
	else:
		st.warning(f"CSV file '{uploaded_file.name}' appears to be empty or missing headers. Treating as plain text.")
		# If no headers, treat as plain text for chunking
		all_content_for_rag_processing.append({
			'text': f"--- File: {uploaded_file.name} ---\n{file_content_raw}",
			'filename': uploaded_file.name,
			'original_content': file_content_raw # Store raw content for embedding if not CSV
		})


def process_uploaded_files(uploaded_files):
	"""Processes uploaded text files, calculates token counts, and updates session state.
	If total token count exceeds a threshold, it processes them with RAG."""
	if not uploaded_files:
		st.session_state.uploaded_file_data = []
		st.session_state.file_token_counts = {}
		st.session_state.rag_context = [] # Clear RAG context
		st.session_state.rag_enabled = False # Disable RAG
		vector_store_manager.clear_index() # Clear RAG index if no files are uploaded
		st.session_state.last_uploaded_filename = None # Clear last uploaded filename
		return

	st.session_state.uploaded_file_data = []
	st.session_state.file_token_counts = {}
	st.session_state.rag_context = [] # Reset RAG context for new uploads
	
	# Initialize RAG components (embedding model only)
	embedding_model.load()
	
	doc_id_counter = 0 # Unique ID for each chunk

	# Clear existing vector store before adding new chunks from current upload
	vector_store_manager.clear_index() 
	# Re-initialize the vector store after clearing it and before adding documents
	vector_store_manager.init_vector_store(dim=embedding_model.model.get_sentence_embedding_dimension())

	all_content_for_rag_processing = [] # List to hold formatted chunks (rows/text) for RAG
	current_uploaded_filenames = [] # To keep track of files in the current upload batch

	for uploaded_file in uploaded_files:
		try:
			file_content_raw = None
			encodings_to_try = ['utf-8', 'big5', 'gbk', 'gb2312', 'latin-1']

			for encoding in encodings_to_try:
				try:
					uploaded_file.seek(0)
					file_content_raw = uploaded_file.read().decode(encoding)
					break  # Exit the loop if decoding is successful
				except UnicodeDecodeError:
					continue  # Try the next encoding

			if file_content_raw is not None:
				tokens = st.session_state.token_encoder.encode(file_content_raw)
				token_count = len(tokens)

				st.session_state.uploaded_file_data.append((uploaded_file.name, file_content_raw))
				st.session_state.file_token_counts[uploaded_file.name] = token_count
				current_uploaded_filenames.append(uploaded_file.name) # Add to current batch
				st.success(f"File '{uploaded_file.name}' uploaded successfully! Tokens: **{token_count}**")

				# --- Conditional processing based on file type ---
				if uploaded_file.name.lower().endswith('.csv'):
					_process_csv_file(uploaded_file, file_content_raw, all_content_for_rag_processing)
				else: # --- Plain text processing ---
					# For non-CSV files, prepend filename and store raw content
					all_content_for_rag_processing.append({
						'text': f"--- File: {uploaded_file.name} ---\n{file_content_raw}",
						'filename': uploaded_file.name,
						'original_content': file_content_raw # Store raw content for embedding
					})
			else:
				st.error(f"Could not decode file '{uploaded_file.name}'. The encoding may be unsupported.")
		except Exception as e:
			st.error(f"Error reading file '{uploaded_file.name}': {e}")

	# Update last_uploaded_filename only if files were actually uploaded in this batch
	if current_uploaded_filenames:
		st.session_state.last_uploaded_filename = current_uploaded_filenames[-1]
	else:
		st.session_state.last_uploaded_filename = None # No files uploaded in this batch

	total_token_count = sum(st.session_state.file_token_counts.values())

	# Define a threshold for RAG processing
	RAG_THRESHOLD = 1500 # You can adjust this value

	if total_token_count > RAG_THRESHOLD:
		st.warning(f"Total tokens ({total_token_count}) exceed the RAG threshold ({RAG_THRESHOLD}). Processing files with RAG...")
		
		# For CSVs, chunks are already formed per row. For text files, we still need text_splitter.
		# Combine all content that needs to be chunked by RecursiveCharacterTextSplitter
		content_for_text_splitter = ""
		for entry in all_content_for_rag_processing:
			# Only add if it's not a CSV row (which is already a "chunk")
			if not entry['filename'].lower().endswith('.csv'):
				content_for_text_splitter += entry['original_content'] + "\n\n"
		
		final_chunks_to_embed = []
		if content_for_text_splitter: # If there's non-CSV text to chunk
			text_splitter = RecursiveCharacterTextSplitter(
				chunk_size=500,  # Smaller chunks for more precise retrieval
				chunk_overlap=100,
				length_function=len, # Use character length for splitting
			)
			chunks_from_text = text_splitter.split_text(content_for_text_splitter)
			for chunk_text in chunks_from_text:
				# We lose specific filename for these chunks if multiple text files were combined.
				# A more advanced solution would track filename per text chunk.
				# For now, assign "multiple_text_files" or similar.
				final_chunks_to_embed.append({
					'text': chunk_text,
					'filename': "multiple_text_files" if len(uploaded_files) > 1 and not uploaded_files[0].name.lower().endswith('.csv') else uploaded_files[0].name
				})
			st.info(f"Splitting non-CSV content into {len(chunks_from_text)} chunks for RAG.")

		# Add the pre-formatted CSV rows as chunks
		for entry in all_content_for_rag_processing:
			if entry['filename'].lower().endswith('.csv'):
				final_chunks_to_embed.append({
					'text': entry['text'], # This is the formatted row string
					'filename': entry['filename']
				})
		
		st.info(f"Total RAG chunks to embed: {len(final_chunks_to_embed)}")

		for i, chunk_info in enumerate(tqdm.tqdm(final_chunks_to_embed, desc="Embedding chunks")): 
			# Embed the content (which is the formatted chunk text)
			vector = embedding_model.embed_text(chunk_info['text'])
			
			# Add to vector store, passing the correct filename
			vector_store_manager.add_document(doc_id_counter, vector, chunk_info['text'], source_filename=chunk_info['filename'])
			doc_id_counter += 1
		
		vector_store_manager.save_metadata()
		st.success(f"All {len(final_chunks_to_embed)} chunks processed and added to vector store for RAG.")

		st.session_state.rag_enabled = True # Indicate that RAG is active
		st.session_state.rag_context = [] # Initialize empty, will be filled on query

	else:
		st.info(f"Total tokens ({total_token_count}) are within the limit ({RAG_THRESHOLD}). No RAG needed for initial processing.")
		st.session_state.rag_enabled = False # Indicate RAG is not active
		st.session_state.rag_context = [] # Ensure RAG context is empty

	
def render_sidebar():
	st.header("Configuration")
	st.markdown("---")

	st.subheader("File Upload")

	# Initialize token encoder in session state if it doesn't exist
	if "token_encoder" not in st.session_state:
		st.session_state.token_encoder = tiktoken.get_encoding("cl100k_base")

	# Initialize rag_context and rag_enabled if not present
	if "rag_context" not in st.session_state:
		st.session_state.rag_context = []
	if "rag_enabled" not in st.session_state:
		st.session_state.rag_enabled = False
	# Initialize last_uploaded_filename if not present
	if 'last_uploaded_filename' not in st.session_state:
		st.session_state.last_uploaded_filename = None

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
			st.session_state.rag_context = [] # Clear RAG context on new chat
			st.session_state.rag_enabled = False # Disable RAG on new chat
			vector_store_manager.clear_index() # Clear RAG index on new chat
			st.session_state.last_uploaded_filename = None # Clear last uploaded filename
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
			st.session_state.rag_context = [] # Clear RAG context on clearing all conversations
			st.session_state.rag_enabled = False # Disable RAG on clearing all conversations
			vector_store_manager.clear_index() # Clear RAG index on clearing all conversations
			st.session_state.last_uploaded_filename = None # Clear last uploaded filename
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
