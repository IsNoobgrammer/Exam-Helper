"""
Utility functions for the Exam-Helper application.
Contains all helper functions for vector stores, LLM operations, and data processing.
"""

import base64
import io
import re
import html
from typing import Dict, Any, List, Optional

import requests as r
import streamlit as st
import google.generativeai as genai
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config_loader import config


def get_vector_store(index_name: str, api_keys: Dict[str, str]) -> PineconeVectorStore:
    """
    Initialize and return a Pinecone vector store.
    
    Args:
        index_name (str): Name of the Pinecone index
        api_keys (Dict[str, str]): Dictionary containing API keys
        
    Returns:
        PineconeVectorStore: Initialized vector store
    """
    pc = Pinecone(api_key=api_keys["pinecone"])
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_keys["google"]
    )
    index = pc.Index(index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)


def get_llm(model: str, api_keys: Dict[str, str]):
    """
    Initialize and return a Google Gemini language model.
    
    Args:
        model (str): Model name to use (e.g., 'models/gemini-1.5-flash')
        api_keys (Dict[str, str]): Dictionary containing API keys
        
    Returns:
        Google Generative AI model instance
    """
    import google.generativeai as genai
    
    # Configure the Gemini API with the provided API key
    genai.configure(api_key=api_keys["google"])
    
    # Set up the model with default temperature
    temperature = config.get("app_config.default_temperature", 0.2)
    
    # Create and return the model
    return genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        },
    )


def get_fallback_llm(api_keys: Optional[Dict[str, str]] = None):
    """
    Get the fallback Gemini LLM.
    
    Args:
        api_keys (Dict[str, str], optional): Dictionary containing API keys
    
    Returns:
        Google Generative AI model instance (Gemini 2.5 Flash Lite)
    """
    import google.generativeai as genai
    
    # Get API keys from config or use provided ones
    if api_keys is None:
        api_keys = config.get_api_keys()
    
    # Configure the Gemini API with the provided API key
    genai.configure(api_key=api_keys["google"])
    
    # Set up the fallback temperature
    temperature = config.get("app_config.fallback_temperature", 0.3)
    
    # Create and return the Gemini model
    return genai.GenerativeModel(
        model_name="models/gemini-2.5-flash-lite",
        generation_config={
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        },
    )


def clean_rag_data(query: str, context: str, llm: Any) -> str:
    """
    Clean and filter RAG data based on the query.
    
    Args:
        query (str): User query
        context (str): Context data to clean
        llm: Language model instance (Gemini model)
        
    Returns:
        str: Cleaned and filtered data
    """
    system_prompt = """
    You are a highly capable Professor of understanding the value and context of both user queries and given data. 
    Your Task for Documents Data is to analyze the list of document's content and properties and find the most important information regarding user's query.
    Your Task for ChatHistory Data is to analyze the given ChatHistory and then provide a ChatHistory relevant to user's query.
    Your Task for Web Data is to analyze the web scraped data then summarize only useful data regarding user's query.
    You Must adhere to User's query before answering.
    
    Output:
        For Document Data
            Conclusion:
                ...
        For ChatHistory Data
                User: ...
                ...
                Assistant: ...
        For Web Data
            Web Scraped Data:
            ...
    """
    
    # Format the prompt for the Gemini model
    prompt = f"""{system_prompt}

{context}
User's query is given below:
{query}
"""
    
    try:
        # Generate response using the Gemini model
        response = llm.generate_content(prompt)
        
        # Extract text from the response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Handle case where response has candidates
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                return ' '.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        
        # Fallback to string representation if structure is unexpected
        return str(response)
    except Exception as e:
        return f"Error processing data: {str(e)}"


def get_llm_data(query: str, llm: Any) -> str:
    """
    Get a response from the LLM based on the query.
    
    Args:
        query (str): User query
        llm: Language model instance (Gemini model)
        
    Returns:
        str: LLM response
    """
    system_prompt = """
    You are a knowledgeable and approachable Computer Science professor with expertise in a wide range of topics.
    Your role is to provide clear, easy, and engaging explanations to help students understand complex concepts.
    When answering:
    - Make it sure to provide the calculations, regarding the solution if there are any.
    - Start with a high-level overview, then dive into details as needed.
    - Use examples, analogies, or step-by-step explanations to clarify ideas.
    - Ensure your answers are accurate, well-structured, and easy to follow.
    - If you don't know the answer, acknowledge it and suggest ways to explore or research further.
    """
    
    # Format the prompt for the Gemini model
    prompt = f"""{system_prompt}

{query}
"""
    
    try:
        # Generate response using the Gemini model
        response = llm.generate_content(prompt)
        
        # Extract text from the response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Handle case where response has candidates
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                return ' '.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        
        # Fallback to string representation if structure is unexpected
        return str(response)
    except Exception as e:
        return f"Error generating LLM response: {str(e)}"


def get_context(
    query: str, 
    use_vector_store: bool,
    vector_store: Optional[PineconeVectorStore], 
    use_web: bool, 
    use_chat_history: bool, 
    llm: Any, 
    llmx: Any, 
    messages: List[Dict[str, str]]
) -> str:
    """
    Retrieve and process context from various sources.
    
    Args:
        query (str): User query
        use_vector_store (bool): Whether to use vector store
        vector_store: Vector store instance
        use_web (bool): Whether to use web search
        use_chat_history (bool): Whether to use chat history
        llm: Primary language model
        llmx: Fallback language model
        messages (List[Dict]): Chat messages history
        
    Returns:
        str: Combined context from all sources
    """
    context = ""
    similarity_k = config.get("app_config.similarity_search_k", 3)
    max_history = config.get("app_config.max_chat_history", 5)
    
    if use_vector_store and vector_store:
        with st.spinner(":green[Extracting Data From VectorStore...]"):
            result = "\n\n".join([
                _.page_content for _ in vector_store.similarity_search(query, k=similarity_k)
            ])
            clean_data = clean_rag_data(query, f"Documents Data \n\n{result}", llmx)
            context += f"Documents Data: \n\n{clean_data}"

    if use_chat_history:
        with st.spinner(":green[Extracting Data From ChatHistory...]"):
            last_messages = messages[:-3][-max_history:]
            chat_history = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in last_messages
            ])
            clean_data = clean_rag_data(
                query, f"\n\nChat History \n\n{chat_history}", llmx
            )
            context += f"\n\nChat History: \n\n{clean_data}"

    try:
        if use_web:
            with st.spinner(":green[Extracting Data From web...]"):
                search = DuckDuckGoSearchRun()
                clean_data = clean_rag_data(query, search.invoke(query), llmx)
                context += f"\n\nWeb Data:\n{clean_data}"
    except Exception as e:
        st.warning(f"Web search failed: {e}")

    if not use_chat_history:
        with st.spinner(":green[Extracting Data From ChatGPT...]"):
            context += f"\n\n LLM Data {get_llm_data(query, llm)}"

    return context


def respond_to_user(query: str, context: str, llm: Any) -> str:
    """
    Generate a response to the user based on the query and context.
    
    Args:
        query (str): User query
        context (str): Context information
        llm: Language model instance (Gemini model)
        
    Returns:
        str: Generated response
    """
    system_prompt = """
    You are a specialized professor of Computer Science Engineering. Your job is to answer the given question based on the following types of context: 

    1. **Web Data**: Information retrieved from web searches.
    2. **Documents Data**: Data extracted from documents (e.g., research papers, reports).
    3. **Chat History**: Previous interactions or discussions in the current session.
    4. **LLM Data**: Insights or completions provided by the language model.

    When answering:
    - Include all important information and key points
    - Provide calculations and detailed explanations where applicable
    - Ensure your response is clear and easy to understand, even for a beginner
    - Format your response with proper markdown for better readability
    - Use bullet points, numbered lists, and code blocks where appropriate
    - If the question is about programming, provide code examples with explanations
    - If the question is conceptual, provide clear definitions and examples
    - If the question is about problem-solving, break down the solution into steps
    """
    
    # Format the prompt for the Gemini model
    prompt = f"""{system_prompt}

Question: {query}

Context:
{context}

Please provide a detailed and well-structured response."""
    
    try:
        # Generate response using the Gemini model
        response = llm.generate_content(prompt)
        
        # Extract text from the response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Handle case where response has candidates
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                return ' '.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        
        # Fallback to string representation if structure is unexpected
        return str(response)
    except Exception as e:
        return f"Error generating response: {str(e)}"


def html_entity_cleanup(text: str) -> str:
    """
    Clean HTML entities from text.
    
    Args:
        text (str): Text containing HTML entities
        
    Returns:
        str: Cleaned text
    """
    replacements = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'"
    }
    
    for entity, replacement in replacements.items():
        text = text.replace(entity, replacement)
    
    return text


def yT_transcript(link: str) -> str:
    """
    Fetch the transcript of a YouTube video.
    
    Args:
        link (str): YouTube video URL
        
    Returns:
        str: Video transcript
    """
    url = config.get("app_config.youtube_transcript_url")
    payload = {"youtube_url": link}
    response = r.post(url, data=payload).text
    
    transcript_segments = re.findall(
        r'class="transcript-segment"[^>]*>\s*([\S ]*?\S)\s*<\/span>', 
        response
    )
    
    return " ".join([html_entity_cleanup(segment) for segment in transcript_segments])


def process_youtube(video_id: str, original_text: str, llmx: Any) -> str:
    """
    Process a YouTube video transcript and answer a query.
    
    Args:
        video_id (str): YouTube video ID
        original_text (str): Original user query
        llmx: Language model instance (Gemini model)
        
    Returns:
        str: Processed response
    """
    transcript = yT_transcript(f"https://www.youtube.com/watch?v={video_id}")
    
    if len(transcript) == 0:
        raise IndexError("No transcript available for this video")
    
    system_prompt = """
You are Explainer Bot, a highly intelligent and efficient assistant designed to analyze YouTube video transcripts and respond comprehensively to user queries. You excel at providing explanations tailored to the user's needs, whether they seek examples, detailed elaboration, or specific insights.

**Persona:**
- You are approachable, insightful, and skilled at tailoring responses to diverse user requests.
- You aim to provide explanations that capture the essence of the video, ensuring a balance between clarity and depth.
- Your tone is clear, neutral, and professional, ensuring readability and understanding for a broad audience.

**Task:**
1. Analyze the provided video transcript, which may contain informal language, repetitions, or filler words. Your job is to:
   - Address the user's specific query, such as providing examples, detailed explanations, or focused insights.
   - Retain the most critical information and adapt your response style accordingly.
2. If the user query contains a YouTube link, do not panic. Use the already provided transcript of the video to answer the query. Ensure your response addresses both the content of the video and any additional parts of the user's query.
3. If the video includes technical or specialized content, provide brief context or explanations where necessary to enhance comprehension.
4. Maintain an organized structure using bullet points, paragraphs, or sections based on the user's query.

**Additional Inputs:**
- When answering:
  - If the user requests examples, include relevant examples or anecdotes from the transcript or generate illustrative examples.
  - If the user requests a detailed explanation, expand on the key points, ensuring no critical information is lost.
  - If the user's query requires a summary, condense the content into a clear, concise explanation while retaining the key messages.
  - Always address the user's specific needs while keeping the overall purpose of the video in focus.

**Output Style:**
- Always respond using **Markdown** format, avoiding LaTeX or any other non-Markdown formatting.
  - Avoid using any LaTeX symbols or complex formatting.
  - Ensure your response is easy to read and compatible with a frontend that supports Markdown.
- Tailor the response to the user's request:
  - Provide examples when explicitly asked or when they are available in the transcript.
  - Offer detailed and comprehensive explanations if required.
  - Keep summaries comprehensive and focused if brevity is requested.
- Use simple, clear sentences to cater to a broad audience.
- Avoid jargon unless it is crucial to the video's context, and provide a brief explanation if used.
- Always answer in English only.

Act as a skilled Professor, ensuring accuracy, brevity, and clarity while retaining the original context and intent of the video. Adjust your tone and structure to match the user's specific query and expectations. If a YouTube link is part of the user query, use the transcript you already have to address the video-related aspects of the question seamlessly.
"""

    # Format the prompt for the Gemini model
    prompt = f"""{system_prompt}

Transcription:
{transcript}

User's Query:
{original_text}
"""
    
    try:
        # Generate response using the Gemini model
        response = llmx.generate_content(prompt)
        
        # Extract text from the response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Handle case where response has candidates
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                return ' '.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
        
        # Fallback to string representation if structure is unexpected
        return str(response)
    except Exception as e:
        return f"Error processing YouTube video: {str(e)}"


def img_to_ques(img: Any, query: str, model: str = "gemini-1.5-flash") -> str:
    """
    Extract a question and relevant information from an image.
    
    Args:
        img: PIL Image object
        query (str): User query
        model (str): Gemini model to use
        
    Returns:
        str: Extracted question and information
    """
    api_keys = config.get_api_keys()
    genai.configure(api_key=api_keys["google_vision"])
    model = genai.GenerativeModel(model)
    
    prompt = f"""Analyze the provided image and the user's query: "{query}". Based on the content of the image:

1. Extract the question from the image, if user wants to ask more questions add it to the Question Section.
2. For any tabular, structured data or MCQ or any other relevant information present in the image, provide it in the "Relevant Information" section.

Format your response as follows:

Question:  
[Generated question based on the image and query]  

Relevant Information:  
[Include any tabular data, key details relevant to solving the problem but it should only come from attached image. If no relevant information is present in image don't add by yourself. 
Ensure structured data is presented in an easily readable format.]
"""
    
    return model.generate_content([prompt, img]).text


class DiagramCheck(BaseModel):
    """Model for diagram requirement check response."""
    requires_diagram: bool = Field(
        ...,
        description="True if the user's question needs a diagram or image for explanation or solution, False otherwise.",
    )
    search_query: str = Field(
        "",
        description="A relevant Google search query to find the required diagram or image, if needed.",
    )


def check_for_diagram(user_query: str, llm: Any) -> DiagramCheck:
    """
    Check if a user query requires a diagram for better explanation.
    
    Args:
        user_query (str): User's question
        llm: Language model instance
        
    Returns:
        DiagramCheck: Result indicating if diagram is needed and search query
    """
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful assistant that analyzes user questions to determine if they require a diagram or image for a better explanation or solution. Your primary goal is to assist with educational and informational queries, especially in the field of Computer Science (CSE).

            - If a diagram/image is needed, set 'requires_diagram' to True and provide a suitable 'search_query' for finding that image on a general search engine.
            - **Give special consideration to diagrams and flowcharts commonly used in Computer Science.** These are often essential for understanding algorithms, data structures, system architectures, and processes. Be lenient when identifying the need for CSE-related diagrams.
            - **The search_query should focus on finding educational, technical, or illustrative content, including relevant CSE diagrams and flowcharts.** It should never explicitly search for or suggest sexually suggestive, explicit, or NSFW (Not Safe For Work) imagery.
            - If a diagram/image is NOT needed, set 'requires_diagram' to False and leave 'search_query' empty.
            - Consider if the question involves:
                - Visualizing structures (e.g., graphs, trees, networks, data structures)
                - Understanding processes (e.g., flowcharts, algorithms, control flow)
                - Comparing visual information
                - Describing layouts, architecture, or designs (especially in a software or system context)
                - Scientific or medical illustrations (e.g., anatomy diagrams, biological processes). These may include representations of the human body for educational purposes, but the focus must remain on the scientific or medical context.
            - **In cases where the user's query might relate to potentially sensitive topics (e.g., human anatomy) or complex CSE topics, be extremely cautious. Prioritize search queries that lead to reputable educational or scientific sources. Avoid any terms that could be interpreted as seeking explicit or inappropriate content.**
            - **Under no circumstances should the 'search_query' include terms like "nude," "naked," "sex," or any other sexually suggestive language.**

            **Examples of Acceptable Queries (for educational/scientific/CSE purposes):**
                - "binary search tree diagram"
                - "linked list vs array visualization"
                - "OSI model flowchart"
                - "CPU scheduling algorithm explained with diagram"
                - "human heart anatomy diagram"
                - "mitosis process illustration"
                - "breast tissue cross-section" (in a medical/biological context)

            **Examples of Unacceptable Queries:**
                - "nude human body"
                - "sexy woman"
                - "breast pictures" (without a clear medical/scientific context)

            Output JSON:
            {{
              "requires_diagram": bool,
              "search_query": str
            }}
            """,
        ),
        ("user", "{user_query}"),
    ])

    # Format the prompt for the Gemini model
    prompt = f"""Analyze the following user query and determine if it requires a diagram or image for better explanation.
    
User Query: {user_query}

Please respond in the following JSON format:
{{
  "requires_diagram": true/false,
  "search_query": "relevant search query if diagram is needed, otherwise empty string"
}}

Guidelines:
- Set 'requires_diagram' to true if the question would benefit from a visual aid (diagram, chart, illustration)
- The search_query should be a concise, relevant search term for finding an appropriate image
- Focus on educational and technical content
- Never include explicit or NSFW terms in the search query
- For CSE topics, be more lenient in suggesting diagrams as they often help in understanding complex concepts

Example response for "Explain how a binary search tree works":
{{
  "requires_diagram": true,
  "search_query": "binary search tree diagram"
}}"""
    
    try:
        # Generate response using the Gemini model
        response = llm.generate_content(prompt)
        
        # Extract text from the response
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Handle case where response has candidates
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                response_text = ' '.join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            else:
                response_text = str(response)
        else:
            response_text = str(response)
        
        # Parse the JSON response
        import json
        try:
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                return DiagramCheck(
                    requires_diagram=result.get('requires_diagram', False),
                    search_query=result.get('search_query', '')
                )
            else:
                # If no JSON found, return default values
                return DiagramCheck(requires_diagram=False, search_query="")
        except json.JSONDecodeError:
            # If JSON parsing fails, return default values
            return DiagramCheck(requires_diagram=False, search_query="")
    except Exception as e:
        # Return default values in case of any error
        return DiagramCheck(requires_diagram=False, search_query="")


def search_images(query: str, num_images: int = 5) -> List[Dict[str, str]]:
    """
    Perform DuckDuckGo image search.
    
    Args:
        query (str): Search query
        num_images (int): Number of images to return
        
    Returns:
        List[Dict]: List of image results
    """
    max_images = config.get("app_config.max_images", 5)
    num_images = min(num_images, max_images)
    
    with DDGS() as ddgs:
        # Get GIF results
        gif_results = [
            dict(text="", title="", img=img['image'], link=img["url"]) 
            for img in ddgs.images(
                query, 
                safesearch='Off', 
                region="en-us", 
                max_results=num_images-2, 
                type_image="gif"
            ) 
            if 'image' in img
        ]
        
        # Get regular image results
        regular_results = [
            dict(text="", title="", img=img['image'], link=img["url"]) 
            for img in ddgs.images(
                query, 
                safesearch='Off', 
                region="en-us", 
                max_results=num_images
            ) 
            if 'image' in img
        ]
        
        return regular_results + gif_results