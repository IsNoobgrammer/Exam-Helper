"""
UI components and styling for the Exam-Helper application.
Contains all Streamlit UI configuration, CSS styling, and component functions.
"""

import streamlit as st
from config_loader import config


def apply_page_config():
    """Apply Streamlit page configuration."""
    app_config = config.get_app_config()
    st.set_page_config(
        page_title=app_config.get("page_title", "Bhala Manus"), 
        page_icon=app_config.get("page_icon", "🌟")
    )


def get_custom_css() -> str:
    """
    Get custom CSS styling for the application.
    
    Returns:
        str: CSS styling string
    """
    ui_config = config.get_ui_config()
    
    return f"""
    <style>
    /* Make header background transparent */
    .stApp > header {{
        background-color: transparent !important;
    }}

    /* Apply animated gradient background */
    .stApp {{
        background: linear-gradient(45deg, #3a5683 10%, #0E1117 45%, #0E1117 55%, #3a5683 90%);
        animation: gradientAnimation {ui_config.get('gradient_animation_duration', '20s')} ease infinite;
        background-size: 200% 200%;
        background-attachment: fixed;
    }}

    /* Keyframes for smooth animation */
    @keyframes gradientAnimation {{
        0% {{
            background-position: 0% 0%;
        }}
        50% {{
            background-position: 100% 100%;
        }}
        100% {{
            background-position: 0% 0%;
        }}
    }}

    /* Main styling */
    .main {{
        font-family: 'Arial', sans-serif;
        background-color: #454545;
        color: #fff;
    }}

    /* Header styling */
    .header {{
        text-align: center;
        color: {ui_config.get('primary_color', '#47fffc')};
        font-size: 36px;
        font-weight: bold;
    }}

    /* Button styling */
    .stButton>button {{
        background-color: {ui_config.get('success_color', '#4CAF50')};
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }}

    /* Text input styling */
    .stTextInput>div>input {{
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid {ui_config.get('success_color', '#4CAF50')};
        padding: 10px;
        font-size: 16px;
    }}

    /* Checkbox styling */
    .stCheckbox>div>label {{
        font-size: 16px;
        color: {ui_config.get('success_color', '#4CAF50')};
    }}

    /* Chat input styling */
    .stChatInput>div>input {{
        background-color: #e8f5e9;
        border: 1px solid #81c784;
    }}

    /* Markdown styling */
    .stMarkdown {{
        font-size: 16px;
    }}

    /* Chat message styling */
    .stChatMessage > div {{
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        font-family: "Arial", sans-serif;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }}

    /* User messages (Dark with 60% opacity, right indent) */
    .stChatMessage > div.user {{
        background-color: rgba(13, 9, 10, 0.6);
        color: #EAF2EF;
        border-left: 4px solid #361F27;
        margin-left: 32px;
        margin-right: 64px;
    }}

    /* Assistant messages (Light Lavender with 40% opacity, left indent) */
    .stChatMessage > div.assistant {{
        background-color: rgba(70, 40, 90, 0.6);
        color: #F5D7FF;
        border-left: 4px solid #BB8FCE;
        margin-left: 64px;
        margin-right: 32px;
    }}

    /* Hover effects for smooth interaction */
    .stChatMessage > div:hover {{
        transform: scale(1.005);
        transition: transform 0.2s ease-in-out;
    }}

    /* Section styling */
    #up, #down, #left, #right {{
        position: fixed;
        z-index: -1;
    }}
    </style>
    """


def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app."""
    css = get_custom_css()
    st.markdown(css, unsafe_allow_html=True)
    
    # Additional HTML sections
    html = """
    <section id="up"></section>
    <section id="down"></section>
    <section id="left"></section>
    <section id="right"></section>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_header():
    """Render the application header."""
    ui_config = config.get_ui_config()
    
    st.markdown(
        f'<div class="header">🌟No Back Abhiyan </div>', 
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="color: {ui_config.get("secondary_color", "#dcfa2f")}; font-size: 18px; text-align: center;">Padh le yaar...</p>',
        unsafe_allow_html=True,
    )


def render_sidebar(models: list, document_indexes: list) -> dict:
    """
    Render the sidebar with configuration options.
    
    Args:
        models (list): Available model options
        document_indexes (list): Available document index options
        
    Returns:
        dict: Configuration values from sidebar
    """
    app_config = config.get_app_config()
    ui_config = config.get_ui_config()
    
    st.sidebar.markdown(
        """<h3 style="color: cyan;">Configuration</h3>""", 
        unsafe_allow_html=True
    )
    
    # Document selection
    index_name = st.sidebar.selectbox(
        "Doc Name", 
        options=document_indexes, 
        index=0, 
        help="Select the name of the Documents to use."
    )
    
    # Google API Key input
    api_key_input = st.sidebar.text_input(
        "Google AI API Key", 
        type="password", 
        help="Enter your Google AI API key or the special passphrase.",
        value=st.session_state.get("api_key_input", ""),
        placeholder="Enter your API key or special passphrase"
    )
    
    # Process API key input
    google_api_key = None
    if api_key_input:
        if api_key_input == "Maybe@123":
            # Use default API key from config for special passphrase
            google_api_key = config.get("api_keys.google")
            st.sidebar.markdown(
                f"<p style='color: {ui_config.get('success_color', '#4CAF50')};'>✅ Using default API key (special access granted)</p>",
                unsafe_allow_html=True,
            )
        else:
            # Use user-provided API key
            google_api_key = api_key_input
            st.sidebar.markdown(
                f"<p style='color: {ui_config.get('success_color', '#4CAF50')};'>✅ Using your API key</p>",
                unsafe_allow_html=True,
            )
        
        # Store the input in session state
        st.session_state["api_key_input"] = api_key_input
        st.session_state["google_api_key"] = google_api_key
    
    # Model selection
    default_model = app_config.get("default_model", models[0] if models else "")
    model_index = models.index(default_model) if default_model in models else 0
    
    model = st.sidebar.selectbox(
        "Select Model",
        options=models,
        index=model_index,
        help="Select the model to use for LLM inference.",
    )
    
    # API Key validation
    if not api_key_input:
        st.sidebar.markdown(
            f"<p style='color: {ui_config.get('warning_color', '#f44336')};'>⚠️ Please enter your Google AI API key or the special passphrase to proceed!</p>",
            unsafe_allow_html=True,
        )
        st.warning("Please enter your Google AI API key or the special passphrase to proceed!")
    
    # Feature toggles
    use_web = st.sidebar.checkbox("Allow Internet Access", value=True)
    use_vector_store = st.sidebar.checkbox("Use Documents", value=True)
    use_chat_history = st.sidebar.checkbox(
        "Use Chat History (Last 2 Chats)", value=False
    )
    
    # Chat history logic
    if use_chat_history:
        use_vector_store, use_web = False, False
    
    # Instructions
    st.sidebar.markdown(
        """
        ---
        **Instructions:**  
        Get your *Google AI API Key*  
        From **[Google AI Studio](https://aistudio.google.com/app/apikey)**
        
        **OR**
        
        Enter the special passphrase for default access
        
        ---
        **Note:** Your API key is required for the application to function. 
        The special passphrase provides temporary access using default credentials.

        --- 
        Kheliye *meating - meeting*
        """
    )
    
    return {
        "index_name": index_name,
        "google_api_key": google_api_key,
        "model": model,
        "use_web": use_web,
        "use_vector_store": use_vector_store,
        "use_chat_history": use_chat_history
    }


def display_chat_history(messages: list):
    """
    Display chat history.
    
    Args:
        messages (list): List of chat messages
    """
    ui_config = config.get_ui_config()
    user_avatar = ui_config.get("user_avatar", "👼")
    assistant_avatar = ui_config.get("assistant_avatar", "🧑‍🏫")
    
    for message in messages:
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar=user_avatar):
                st.write(message["content"])
        else:
            with st.chat_message(message["role"], avatar=assistant_avatar):
                st.write(message["content"])


def show_success_message(message: str):
    """
    Show a success message.
    
    Args:
        message (str): Success message to display
    """
    st.success(message)


def show_warning_message(message: str):
    """
    Show a warning message.
    
    Args:
        message (str): Warning message to display
    """
    st.warning(message)


def show_info_message(message: str):
    """
    Show an info message.
    
    Args:
        message (str): Info message to display
    """
    st.info(message)