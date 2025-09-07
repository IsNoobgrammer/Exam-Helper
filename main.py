"""
Main application file for Exam-Helper: Bhala Manus - No BackLog Abhiyan
A comprehensive AI-powered study companion for Computer Science students.
"""

import base64
import io
import re
from typing import Dict, Any, List, Optional

import streamlit as st
from PIL import Image
from st_multimodal_chatinput import multimodal_chatinput
from streamlit_carousel import carousel

from config_loader import config
from ui_components import (
    apply_page_config, 
    apply_custom_styling, 
    render_header, 
    render_sidebar,
    display_chat_history,
    show_success_message,
    show_warning_message,
    show_info_message
)
import utils


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = "El Gamal"


def setup_vector_store_and_llm(sidebar_config: Dict[str, Any]) -> tuple:
    """
    Set up vector store and language model based on configuration.
    
    Args:
        sidebar_config (Dict): Configuration from sidebar
        
    Returns:
        tuple: (vector_store, llm, llmx) instances
    """
    google_api_key = sidebar_config["google_api_key"]
    index_name = sidebar_config["index_name"]
    model = sidebar_config["model"]
    
    # Get API keys with the user-provided or default Google API key
    api_keys = config.get_api_keys(google_api_key)
    vector_store = None
    llm = None
    
    # Initialize vector store
    if "vector_store" not in st.session_state and google_api_key:
        vector_store = utils.get_vector_store(index_name, api_keys)
        st.session_state["vector_store"] = vector_store
        st.session_state["index_name"] = index_name
        show_success_message(
            f"Successfully connected to the Vector Database: {index_name}! Let's go..."
        )
    else:
        vector_store = st.session_state.get("vector_store")

    # Update vector store if index changed
    if ("index_name" in st.session_state and 
        st.session_state["index_name"] != index_name and google_api_key):
        vector_store = utils.get_vector_store(index_name, api_keys)
        st.session_state["vector_store"] = vector_store
        st.session_state["index_name"] = index_name
        show_success_message(
            f"Successfully connected to the Vector Database: {index_name}! Let's go..."
        )

    # Initialize LLM
    if google_api_key:
        if "llm" not in st.session_state:
            llm = utils.get_llm(model, api_keys)
            st.session_state["llm"] = llm
            st.session_state["model"] = model
            st.session_state["api_key"] = google_api_key
        else:
            llm = st.session_state["llm"]

        # Update LLM if API key or model changed
        if ("api_key" in st.session_state and "model" in st.session_state):
            if (google_api_key != st.session_state["api_key"] or 
                model != st.session_state["model"]):
                llm = utils.get_llm(model, api_keys)
                st.session_state["llm"] = llm
                st.session_state["model"] = model
                st.session_state["api_key"] = google_api_key

    # Fallback model with API keys
    llmx = utils.get_fallback_llm(api_keys)
    
    return vector_store, llm, llmx


def extract_youtube_video_id(text: str) -> str:
    """
    Extract YouTube video ID from text.
    
    Args:
        text (str): Text containing potential YouTube URL
        
    Returns:
        str: Video ID if found, empty string otherwise
    """
    patterns = [
        r"(?:https://www\.youtube\.com/watch\?v=([^&\n]+))",
        r"(?:https://youtu.be/([^\?\n]+))"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
    
    return ""


def process_user_input(
    user_inp: Dict[str, Any], 
    vector_store: Any, 
    llm: Any, 
    llmx: Any, 
    sidebar_config: Dict[str, Any]
) -> None:
    """
    Process user input and generate response.
    
    Args:
        user_inp (Dict): User input from multimodal chat
        vector_store: Vector store instance
        llm: Primary language model
        llmx: Fallback language model
        sidebar_config (Dict): Sidebar configuration
    """
    # Check for duplicate query
    if user_inp == st.session_state["last_query"]:
        st.stop()
    else:
        st.session_state["last_query"] = user_inp

    video_id = ""
    question = ""
    
    # Process image input
    if user_inp["images"]:
        b64_image = user_inp["images"][0].split(",")[-1]
        image = Image.open(io.BytesIO(base64.b64decode(b64_image)))
        try:
            question = utils.img_to_ques(image, user_inp["text"])
        except Exception:
            question = utils.img_to_ques(image, user_inp["text"], "gemini-2.0-flash-exp")
        
        # Extract video ID from image text
        video_id = extract_youtube_video_id(user_inp["text"])
        user_inp["text"] = ""
    
    # Extract video ID from text if not found in image
    if not video_id:
        video_id = extract_youtube_video_id(user_inp["text"])
    
    # Add user message to chat history
    full_query = question + user_inp["text"]
    st.session_state.messages.append({"role": "user", "content": full_query})
    
    # Check for diagram requirement
    with st.spinner(":green[Checking Requirements For Image]"):
        diagram_required = utils.check_for_diagram(full_query, llmx)
    
    if diagram_required.requires_diagram:
        with st.spinner(":green[Generating Diagram]"):
            try:
                images = utils.search_images(diagram_required.search_query, 5)
                if images:
                    carousel(images, fade=True, wrap=True, interval=999000)
            except Exception as e:
                show_warning_message(f"Unable to Generate Diagram Due to Error: {e}")
    else:
        show_info_message("No Diagram Required For This Query")
    
    # Process YouTube video if found
    if video_id:
        with st.spinner(":green[Processing Youtube Video]"):
            show_success_message(f"!! Youtube Link Found:- {video_id} , Summarizing Video")
            try:
                yt_response = utils.process_youtube(video_id, full_query, llmx)
            except Exception as e:
                yt_response = f"Unable to Process Youtube Video Due to Transcript not available Error: {e}"
            
            st.session_state.messages.append({"role": "assistant", "content": yt_response})
            
            # Display conversation
            with st.chat_message("user", avatar="👼"):
                st.write(full_query)
            with st.chat_message("assistant", avatar="🧑‍🏫"):
                st.write(yt_response)
    
    # Process regular query if no video
    if not video_id:
        context = utils.get_context(
            full_query,
            sidebar_config["use_vector_store"],
            vector_store,
            sidebar_config["use_web"],
            sidebar_config["use_chat_history"],
            llm,
            llmx,
            st.session_state.messages,
        )
        
        with st.spinner(":green[Combining jhol jhal...]"):
            assistant_response = utils.respond_to_user(full_query, context, llm)
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Display conversation
        with st.chat_message("user", avatar="👼"):
            st.write(full_query)
        with st.chat_message("assistant", avatar="🧑‍🏫"):
            st.write(assistant_response)


def main():
    """Main application function."""
    # Initialize page configuration
    apply_page_config()
    apply_custom_styling()
    
    # Render header
    render_header()
    
    # Get configuration
    models = config.get_models()
    document_indexes = config.get_document_indexes()
    
    # Render sidebar and get configuration
    sidebar_config = render_sidebar(models, document_indexes)
    
    # Initialize session state
    initialize_session_state()
    
    # Set up vector store and LLM
    vector_store, llm, llmx = setup_vector_store_and_llm(sidebar_config)
    
    # Display chat history
    display_chat_history(st.session_state.messages)
    
    # Main chat interaction
    if sidebar_config.get("google_api_key"):
        with st.container():
            user_inp = multimodal_chatinput()
        
        with st.container():
            if user_inp:
                process_user_input(
                    user_inp, 
                    vector_store, 
                    llm, 
                    llmx, 
                    sidebar_config
                )


if __name__ == "__main__":
    main()