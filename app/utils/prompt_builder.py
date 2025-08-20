import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage

def build_prompt(system_prompt: str, selected_language: str, show_cot: bool, reasoning_effort: str, chat_history: list, rag_context: str = None) -> list:
    """
    Constructs the LLM prompt based on session state variables.

    Parameters
    ----------
    system_prompt : str
        The base system prompt from the configuration.
    selected_language : str
        The user's selected language ('en' for English, 'zh-tw' for Traditional Chinese).
    show_cot : bool
        Whether to include Chain-of-Thought instructions.
    reasoning_effort : str
        The selected reasoning effort ('low', 'medium', 'high').
    chat_history : list
        The current chat history.
    rag_context : str, optional
        The retrieved context from RAG, if available. Defaults to None.

    Returns
    -------
    list
        A list of Langchain messages representing the constructed prompt.
    """
    language_instruction = "Respond in English."
    if selected_language == "zh-tw":
        language_instruction = "Respond in Traditional Chinese."
    
    # Combine the base system prompt with language instructions
    full_system_prompt = f"{system_prompt} {language_instruction}"

    if show_cot:
        if selected_language == "zh-tw":
            full_system_prompt += "\n\n請先以 '思考過程：' 開頭解釋你的推理和思考流程，然後再以 '最終答案：' 開頭給出最終答案。"
        else:
            full_system_prompt += "\n\nFirst, explain your reasoning and thought process starting with 'Thought:'. Then, provide your final answer starting with 'Answer:'."

    # Add reasoning effort instruction to the system prompt
    full_system_prompt += f"\nMust use this reasoning_effort: {reasoning_effort};"

    # Add RAG context to the system prompt if available
    if rag_context:
        full_system_prompt += f"\n\nHere is some relevant context from uploaded documents:\n{rag_context}\n\n"

    prompt = [SystemMessage(content=full_system_prompt)]

    # Only include the user's *original* message, as the RAG context
    # is now part of the system prompt in this architecture.
    # We also need to handle the case where combined_input was put into chat_history.
    # We should ensure the last message in chat_history is the actual user query
    # without the appended file content/RAG context if RAG was used.
    # For now, we extend chat_history from the end as before.
    # A more robust solution might involve passing only the user's *original* input
    # to prompt_builder and then adding RAG context separately.

    # Let's adjust this to ensure only the original user query is part of chat_history
    # and the RAG context is separate.
    # This requires changing how `combined_input` is handled in chat_area.py as well.

    # If rag_context is provided, it means combined_input in chat_area.py
    # would have included the RAG context. To prevent redundancy and maintain
    # a clean chat history for the LLM, we should ensure the last user message
    # in chat_history only contains the original user query.

    # For this current prompt_builder, if rag_context is provided,
    # the last user message in chat_history might contain the combined_input
    # (user_input + rag_context). This will lead to duplicate context.
    # A better approach would be to send the original user_input to the LLM directly
    # and add rag_context to the system prompt, as we're doing here.
    # So, the `chat_history` passed to `build_prompt` should ideally
    # *not* contain the `rag_context` already appended to the last user message.

    # Let's assume chat_history only contains the conversational turns
    # without the RAG context appended to the *latest* user message,
    # and the RAG context is passed separately.
    
    # The last message in chat_history should be the original user's message
    # if rag_context was passed separately.
    # So, chat_history[-st.session_state.history_length:] remains valid
    # for previous conversation turns.
    
    # IMPORTANT: The user_input sent to the chat_message("user").write(combined_input)
    # in chat_area.py needs to be reconsidered if rag_context is being sent separately
    # to the LLM via the system prompt.
    # If `combined_input` already includes the RAG context, and `rag_context`
    # is also passed here, it will be duplicated.

    # To avoid duplication:
    # 1. In chat_area.py, when `rag_context` is present, `st.session_state.chat_history.append(HumanMessage(content=user_input))` should happen.
    # 2. The display in `st.chat_message("user").write()` should still show the combined input.
    # This is a subtle but important distinction for what the LLM *sees* vs. what the user *sees*.

    # For now, based on the current `chat_area.py` which appends `combined_input`
    # to chat_history, we need to ensure that if `rag_context` is used, the last
    # message in chat_history is modified to *only* contain the original user input
    # for the LLM's consumption. This is a more complex change.

    # A simpler fix for *this* prompt_builder is to ensure that `rag_context` is *only*
    # added if the calling function (chat_area) *doesn't* already combine it into the
    # user's message in chat_history.
    # Given the previous chat_area.py already combines, we need to adapt.

    # Let's revisit chat_area.py to ensure the chat_history properly handles this.
    # If rag_context is used, the HumanMessage added to chat_history should only be user_input.
    # The `combined_input` is just for display.

    # Assuming chat_history contains the actual conversational turns without RAG context
    # appended to the latest user message when RAG is active.
    # This means the last item in `chat_history` should be `HumanMessage(content=user_input)`.

    # Based on the current chat_area.py, `st.session_state.chat_history.append(HumanMessage(content=combined_input))`
    # means the RAG context (or raw file content) IS already in the chat history.
    # So, passing `rag_context` here would duplicate it if it was added to `combined_input`.

    # The most straightforward way, given existing `chat_area.py`, is to:
    # 1. Ensure `process_uploaded_files` populates `st.session_state.rag_context` with *only* the RAG results.
    # 2. `chat_area.py` will decide whether to append raw files OR use RAG context.
    #    If RAG context is used, it should be appended to the user_input, and that combined
    #    string is put into `chat_history`.
    #    The `rag_context` parameter here in `build_prompt` becomes redundant if
    #    it's already in the last `HumanMessage` of `chat_history`.

    # Let's adjust `build_prompt` to simply *always* use chat_history as is,
    # assuming chat_area has done the work of composing the last HumanMessage.
    # The `rag_context` parameter in `build_prompt` is then primarily for the *system prompt*
    # to give the LLM explicit instruction about *how* to use the provided context.

    # The issue arises if `combined_input` (which is put into chat_history)
    # includes the RAG context AND the system prompt also includes it.
    # To avoid this, `rag_context` should be used to enrich the *system prompt*,
    # and the user's message should ideally be the original user query.

    # For the current structure of `chat_area.py`, where `combined_input`
    # (user_input + RAG/file content) is added to `chat_history`,
    # the `rag_context` parameter in `build_prompt` is somewhat redundant
    # for *content passing*. It would primarily be for *instruction*.

    # Let's modify chat_area to pass the *original* user input for the HumanMessage,
    # and then pass the `rag_context` (if applicable) separately to prompt_builder.
    # This makes the prompt construction cleaner and avoids duplication.

    # REVISED PLAN:
    # 1. In chat_area.py:
    #    a. When user_input is received:
    #       - Store original `user_input_for_llm = user_input`.
    #       - Construct `display_input = user_input` (what the user sees).
    #       - If `st.session_state.rag_context` exists:
    #           - Add `formatted_rag_context` to `display_input`.
    #           - Pass `formatted_rag_context` to `prompt_builder.build_prompt` via `rag_context` parameter.
    #       - Else if `st.session_state.uploaded_file_data` exists:
    #           - Add `all_file_contents` to `display_input`.
    #           - Pass `all_file_contents` to `prompt_builder.build_prompt` via `rag_context` parameter (as it's essentially the "context" for this turn).
    #       - Add `HumanMessage(content=user_input_for_llm)` to `st.session_state.chat_history`.
    #       - `st.chat_message("user").write(display_input)`
    # 2. In prompt_builder.py:
    #    a. `build_prompt` receives `rag_context`.
    #    b. It adds `rag_context` to the `full_system_prompt`.
    #    c. `prompt.extend(chat_history[-st.session_state.history_length:])` remains, which will now contain
    #       the actual user queries (not the combined display strings).

    # This ensures the LLM receives the RAG context (or raw file content) via the system prompt (which
    # is generally better for providing context/instructions) and the chat history only contains
    # the clean user/AI turns.

    # Let's apply this logic to the provided files.

    prompt = [SystemMessage(content=full_system_prompt)]
    prompt.extend(chat_history[-st.session_state.history_length:])
    
    return prompt
