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
        The current chat history (should contain only original user/AI messages).
    rag_context : str, optional
        The retrieved context from RAG or raw file content, if available. Defaults to None.

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
    
    # Extend with the chat history, which now only contains clean conversational turns
    prompt.extend(chat_history[-st.session_state.history_length:])
    
    return prompt
