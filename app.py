import os
import sys
import tempfile
import json
import uuid
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from PyPDF2 import PdfReader
import streamlit as st
import spacy
import torch
import openai
from dotenv import load_dotenv
import base64

# You can add your API key directly in the code
# Replace with your actual OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Load environment variables from .env file (optional, as fallback)
load_dotenv()

# Import transformers components
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Import additional components for multimodal support
from PIL import Image
import cv2
from io import BytesIO

# For audio processing
try:
    from streamlit_mic_recorder import mic_recorder
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False

# For visualization
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Fix for asyncio "no running event loop" error
if sys.platform == 'win32':
    import asyncio
    try:
        from asyncio import WindowsSelectorEventLoopPolicy
        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    except ImportError:
        pass
    
    # Make sure we have a proper event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# Fix torch path warnings
try:
    # Force initialization of torch classes to prevent path errors
    torch._C._jit_set_profiling_mode(False)
except:
    pass

# Set the page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🌟",
    layout="wide"
)

# Initialize session state variables
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'chunk_metadata' not in st.session_state:
    st.session_state.chunk_metadata = []
if 'document_embeddings' not in st.session_state:
    st.session_state.document_embeddings = None
if 'document_sections' not in st.session_state:
    st.session_state.document_sections = {}
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'llm_tokenizer' not in st.session_state:
    st.session_state.llm_tokenizer = None
if 'llm_pipeline' not in st.session_state:
    st.session_state.llm_pipeline = None
if 'last_question_embedding' not in st.session_state:
    st.session_state.last_question_embedding = None
if 'image_processors' not in st.session_state:
    st.session_state.image_processors = {}
if 'images' not in st.session_state:
    st.session_state.images = []
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = []
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'visualization_data' not in st.session_state:
    st.session_state.visualization_data = None
if 'embedding_2d' not in st.session_state:
    st.session_state.embedding_2d = None

# App title and description
st.title("🌟 Multimodal RAG System")
st.markdown("""
This app uses Retrieval-Augmented Generation to answer questions about multimodal content.
Upload text documents, images, and audio to create a comprehensive knowledge base.
""")

# Initialize OpenAI client from hardcoded key
try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # Test with a simple request
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=["Test connection"]
    )
    st.session_state.openai_client = client
except Exception as e:
    # Fallback to environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            st.session_state.openai_client = client
        except:
            pass

# Sidebar for config settings
with st.sidebar:
    st.header("Configuration Settings")
    
    llm_model_name = st.selectbox(
        "Language Model",
        ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
         "facebook/opt-125m",
         "databricks/dolly-v2-3b"],
        index=0,
        help="Select a language model for answer generation"
    )
    
    # Retrieval settings
    st.header("Retrieval Settings")
    
    top_k = st.slider(
        "Number of chunks to retrieve", 
        min_value=3, 
        max_value=15, 
        value=6,
        help="More chunks provide more context but may include irrelevant information"
    )
    
    semantic_weight = st.slider(
        "Semantic similarity weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7,
        help="Balance between semantic similarity and keyword matching"
    )
    
    # OpenAI API status
    st.header("OpenAI API Status")
    if st.session_state.openai_client:
        st.success("✅ OpenAI API connected")
    else:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            try:
                client = openai.OpenAI(api_key=api_key)
                # Test with a simple request
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=["Test connection"]
                )
                st.session_state.openai_client = client
                st.success("✅ API key is valid")
            except Exception as e:
                st.error(f"❌ Invalid API key: {str(e)}")

# Load SpaCy model for NER
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.info("Downloading SpaCy model (this will only happen once)...")
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Load language model for answer generation
@st.cache_resource
def load_language_model(model_name):
    """Load the language model and tokenizer for answer generation"""
    st.info(f"Loading language model {model_name}... (this may take a few minutes)")
    
    # Set up device - use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load model and tokenizer with lower precision to save memory
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"  # Automatically handle device placement
        )
        
        # Create generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        
        return tokenizer, model, pipe
    except Exception as e:
        st.error(f"Error loading language model: {str(e)}")
        st.info("Falling back to rule-based answer generation.")
        return None, None, None

# Image processing functions with model caching
@st.cache_resource
def load_image_captioning_model():
    """Load and cache the image captioning model"""
    from transformers import pipeline
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Process image with the selected vision model
def process_image(image, processor_type="caption"):
    """Process an image with the selected processor"""
    if processor_type == "ocr":
        # Use OCR to extract text from image
        try:
            # Try using TrOCR model from Hugging Face
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            if "trocr_model" not in st.session_state.image_processors:
                with st.spinner("Loading TrOCR model for text recognition..."):
                    try:
                        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
                        
                        # Store in session state
                        st.session_state.image_processors["trocr_model"] = model
                        st.session_state.image_processors["trocr_processor"] = processor
                    except Exception as e:
                        st.error(f"Error loading TrOCR model: {str(e)}")
                        return f"Failed to load OCR model: {str(e)}"
            else:
                processor = st.session_state.image_processors["trocr_processor"]
                model = st.session_state.image_processors["trocr_model"]
            
            # Process the image
            pixel_values = processor(image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return extracted_text
        except Exception as e:
            st.error(f"Error in OCR: {str(e)}")
            return "OCR processing failed."
    
    elif processor_type == "caption":
        # Use image captioning model
        try:
            if "image_captioner" not in st.session_state.image_processors:
                with st.spinner("Loading image captioning model..."):
                    st.session_state.image_processors["image_captioner"] = load_image_captioning_model()
            
            captioner = st.session_state.image_processors["image_captioner"]
            caption = captioner(image)[0]["generated_text"]
            return caption
        except Exception as e:
            st.error(f"Error in image captioning: {str(e)}")
            return "Image captioning failed."
    
    elif processor_type == "classification":
        # Use image classification model
        try:
            from transformers import pipeline
            if "image_classifier" not in st.session_state.image_processors:
                with st.spinner("Loading image classification model..."):
                    st.session_state.image_processors["image_classifier"] = pipeline("image-classification")
            
            classifier = st.session_state.image_processors["image_classifier"]
            predictions = classifier(image)
            
            # Format results
            result = "Image contains: "
            for pred in predictions[:5]:  # Top 5 predictions
                result += f"{pred['label']} ({pred['score']:.2f}), "
            
            return result.rstrip(", ")
        except Exception as e:
            st.error(f"Error in image classification: {str(e)}")
            return "Image classification failed."
    
    return "No processing applied."

# Updated audio processing with OpenAI Whisper
def process_audio_with_whisper(audio_file, client):
    """
    Process audio with OpenAI Whisper API for high-quality transcription
    
    Parameters:
    - audio_file: Path to the audio file
    - client: OpenAI client instance
    
    Returns:
    - Transcribed text
    """
    try:
        # Open the audio file
        with open(audio_file, "rb") as file:
            # Call the OpenAI Whisper API
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )
            return transcription.text
    except Exception as e:
        st.error(f"Error in Whisper transcription: {str(e)}")
        return f"Transcription failed: {str(e)}"

# Enhanced audio tab with Whisper integration
def build_audio_tab():
    st.header("Add Audio Content")
    
    if st.session_state.openai_client is None:
        st.warning("OpenAI API key is required for audio processing. Please enter your API key in the sidebar.")
        return
    
    # Create columns for the two options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Record Audio")
        
        # Check if streamlit-mic-recorder is available
        try:
            from streamlit_mic_recorder import mic_recorder
            
            # Record audio
            audio_bytes = mic_recorder(
                key="recorder", 
                start_prompt="Start Recording", 
                stop_prompt="Stop Recording",
                use_container_width=True
            )
            
            if audio_bytes:
                # Display the recorded audio
                st.audio(audio_bytes["bytes"])
                
                # Save to temporary file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_file.write(audio_bytes["bytes"])
                    temp_path = temp_file.name
                
                try:
                    # Process with Whisper
                    with st.spinner("Transcribing with OpenAI Whisper..."):
                        transcription = process_audio_with_whisper(
                            temp_path, 
                            st.session_state.openai_client
                        )
                        
                    # Display transcription
                    st.subheader("Transcription:")
                    st.write(transcription)
                    
                    # Add to knowledge base
                    if transcription and st.button("Add to Knowledge Base", key="add_recording"):
                        # Create a new chunk
                        new_chunk_id = len(st.session_state.text_chunks)
                        st.session_state.text_chunks.append(f"Audio transcription: {transcription}")
                        st.session_state.chunk_metadata.append({
                            "source_type": "audio",
                            "extraction_method": "whisper",
                            "chunk_type": "audio_transcription"
                        })
                        
                        # Create embedding
                        if st.session_state.openai_client is not None:
                            new_embedding = st.session_state.openai_client.embeddings.create(
                                model="text-embedding-ada-002",
                                input=[f"Audio transcription: {transcription}"]
                            ).data[0].embedding
                            
                            if st.session_state.document_embeddings is not None:
                                st.session_state.document_embeddings = np.vstack([
                                    st.session_state.document_embeddings, 
                                    [new_embedding]
                                ])
                            else:
                                st.session_state.document_embeddings = np.array([new_embedding])
                            
                            # Store audio info
                            st.session_state.audio_files.append({
                                "audio_bytes": audio_bytes["bytes"],
                                "transcription": transcription
                            })
                            
                            st.success("✅ Audio transcription added to knowledge base!")
                        
                finally:
                    # Clean up temp file
                    os.unlink(temp_path)
                    
        except ImportError:
            st.warning("The streamlit-mic-recorder component is not installed. Install with: pip install streamlit-mic-recorder")
            st.info("You can still upload audio files below.")
    
    with col2:
        st.subheader("Upload Audio File")
        
        # File uploader for audio
        audio_file = st.file_uploader(
            "Upload an audio file", 
            type=["wav", "mp3", "m4a", "ogg"], 
            key="audio_upload"
        )
        
        if audio_file is not None:
            # Play the uploaded audio
            st.audio(audio_file)
            
            # Button to transcribe
            if st.button("Transcribe with Whisper"):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.name.split(".")[-1]}') as temp_file:
                    temp_file.write(audio_file.getvalue())
                    temp_path = temp_file.name
                
                try:
                    # Process with Whisper
                    with st.spinner("Transcribing with OpenAI Whisper..."):
                        transcription = process_audio_with_whisper(
                            temp_path, 
                            st.session_state.openai_client
                        )
                    
                    # Display transcription
                    st.subheader("Transcription:")
                    st.write(transcription)
                    
                    # Add to knowledge base
                    if transcription and st.button("Add to Knowledge Base", key="add_uploaded_audio"):
                        # Create a new chunk
                        new_chunk_id = len(st.session_state.text_chunks)
                        st.session_state.text_chunks.append(f"Audio transcription: {transcription}")
                        st.session_state.chunk_metadata.append({
                            "source_type": "audio",
                            "file_name": audio_file.name,
                            "extraction_method": "whisper",
                            "chunk_type": "audio_transcription"
                        })
                        
                        # Create embedding
                        if st.session_state.openai_client is not None:
                            new_embedding = st.session_state.openai_client.embeddings.create(
                                model="text-embedding-ada-002",
                                input=[f"Audio transcription: {transcription}"]
                            ).data[0].embedding
                            
                            if st.session_state.document_embeddings is not None:
                                st.session_state.document_embeddings = np.vstack([
                                    st.session_state.document_embeddings, 
                                    [new_embedding]
                                ])
                            else:
                                st.session_state.document_embeddings = np.array([new_embedding])
                            
                            # Store audio info
                            st.session_state.audio_files.append({
                                "file_name": audio_file.name,
                                "transcription": transcription
                            })
                            
                            st.success("✅ Audio transcription added to knowledge base!")
                finally:
                    # Clean up temp file
                    os.unlink(temp_path)
    
    # Display existing audio transcriptions
    if st.session_state.audio_files:
        st.header("Audio Transcriptions in Knowledge Base")
        for i, audio_data in enumerate(st.session_state.audio_files):
            with st.expander(f"Audio {i+1}: {audio_data.get('file_name', 'Recorded Audio')}"):
                if "audio_bytes" in audio_data:
                    st.audio(audio_data["audio_bytes"])
                st.markdown(f"**Transcription:**\n\n{audio_data['transcription']}")

# Add required dependencies to requirements list
def display_audio_requirements():
    st.info("""
    To use the audio features, you need these additional libraries:
    
    ```
    pip install openai streamlit-mic-recorder
    ```
    
    For the OpenAI Whisper API (best accuracy):
    - Make sure your OpenAI API key has billing enabled
    - Whisper is charged at $0.006 per minute for audio files
    """)
    
    # Optional: One-click installation
    if st.button("Install Audio Dependencies"):
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "streamlit-mic-recorder", "openai"],
                capture_output=True,
                text=True
            )
            st.success("Dependencies installed! Please restart the app.")
            st.code(result.stdout)
        except Exception as e:
            st.error(f"Installation failed: {str(e)}")
            st.info("Please install the packages manually using pip.")

# IMPROVED: Document structure extraction for narrative text
def extract_document_structure(text):
    """
    Extract document structure with special handling for narrative texts like stories
    that may not have traditional section numbering
    """
    sections = {}
    
    # Early return for empty text
    if not text or len(text.strip()) == 0:
        return sections
    
    try:
        # First, try to find story titles (e.g., "Amarok the lone wolf")
        # Look for lines that are likely titles (capitalized words at start of paragraphs)
        story_title_pattern = r'(?:^|\n\s*\n)([A-Z][a-z]+(?:\s+[a-z]*\s+)*[A-Z][a-z]+)'
        story_matches = list(re.finditer(story_title_pattern, text))
        
        # If we found potential story titles
        if story_matches:
            for i, match in enumerate(story_matches):
                title = match.group(1).strip()
                if len(title) > 0 and len(title) < 50:  # Reasonable title length
                    section_num = str(i + 1)
                    start_pos = match.start()
                    
                    # Determine end position (next story or end of text)
                    end_pos = len(text)
                    if i < len(story_matches) - 1:
                        end_pos = story_matches[i+1].start()
                    
                    # Extract content
                    content = text[start_pos:end_pos].strip()
                    
                    # Store the section
                    sections[section_num] = {
                        'title': title,
                        'start': start_pos,
                        'level': 1,
                        'content': content
                    }
        
        # If no story titles found, fallback to using paragraphs
        if not sections:
            paragraphs = re.split(r'\n\s*\n', text)
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 50:  # Only use substantial paragraphs
                    section_id = str(i + 1)
                    # Use first few words as a title
                    words = para.strip().split()
                    title = " ".join(words[:min(5, len(words))]) + "..."
                    
                    sections[section_id] = {
                        'title': title,
                        'start': text.find(para),
                        'level': 1,
                        'content': para
                    }
        
        # If still no sections, use the whole document
        if not sections:
            sections["1"] = {
                'title': "Entire Document",
                'start': 0,
                'level': 1,
                'content': text
            }
                    
    except Exception as e:
        print(f"Error in extract_document_structure: {str(e)}")
        
        # Fallback: If all else fails, treat the entire text as one section
        sections["1"] = {
            'title': "Entire Document",
            'start': 0,
            'level': 1,
            'content': text
        }
    
    return sections

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name
    
    text = ""
    pages = []
    
    try:
        pdf_reader = PdfReader(temp_path)
        
        # Extract text from each page with page numbers
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
                pages.append({"page_num": i+1, "text": page_text})
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
    
    # Extract document structure
    document_structure = extract_document_structure(text)
    
    return text, pages, document_structure

# IMPROVED: Split text into chunks with semantic boundaries
def improved_split_text(pages, document_structure):
    """Split text into chunks with semantic boundaries for better retrieval"""
    chunks = []
    chunk_metadata = []
    
    # If document has stories/sections, use those for chunking
    if document_structure and len(document_structure) > 0:
        for section_num, section_info in document_structure.items():
            content = section_info.get('content', '')
            title = section_info.get('title', '')
            
            if not content:
                continue
                
            # For each section, create narrative-aware chunks
            # Split into semantic units (paragraphs or major sentences)
            semantic_units = re.split(r'(?<=\.)\s+(?=[A-Z])', content)
            
            current_chunk = ""
            for unit in semantic_units:
                # If adding this unit would make chunk too large, save current chunk and start new one
                if len(current_chunk) + len(unit) > 800 and len(current_chunk) > 200:
                    chunks.append(current_chunk.strip())
                    chunk_metadata.append({
                        "section": section_num,
                        "section_title": title,
                        "chunk_type": "semantic_unit",
                        "source_type": "pdf"
                    })
                    
                    # Start new chunk with title context for better retrieval
                    current_chunk = f"{title} - " + unit
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += " " + unit
                    else:
                        # First unit in section, include title context
                        current_chunk = f"{title} - " + unit
            
            # Don't forget the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                chunk_metadata.append({
                    "section": section_num,
                    "section_title": title,
                    "chunk_type": "semantic_unit",
                    "source_type": "pdf"
                })
    else:
        # Fallback: process page by page if no document structure
        for page in pages:
            page_text = page["text"]
            page_num = page["page_num"]
            
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', page_text)
            
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                    continue
                
                # Add paragraph as a chunk
                chunks.append(paragraph.strip())
                chunk_metadata.append({
                    "page": page_num,
                    "paragraph": para_idx,
                    "chunk_type": "paragraph",
                    "source_type": "pdf"
                })
    
    return chunks, chunk_metadata

# Create document embeddings with OpenAI
def create_document_embeddings(text_chunks, client):
    """Create embeddings for all text chunks using OpenAI API"""
    if client is None:
        st.error("OpenAI client not initialized. Check your API key.")
        return None
    
    embeddings = []
    
    with st.spinner("Creating embeddings with OpenAI API (this may take a moment)..."):
        # Process in batches to avoid rate limits
        batch_size = 20
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            
            try:
                # Call OpenAI embedding API
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
                return None
    
    # Create 2D visualization of embeddings
    try:
        create_embedding_visualization(embeddings, text_chunks)
    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
    
    return np.array(embeddings)

# Create 2D visualization of embeddings using PCA
def create_embedding_visualization(embeddings, text_chunks):
    """Create and store a 2D visualization of the embeddings"""
    if len(embeddings) < 5:
        return
    
    # Use PCA for dimensionality reduction to 2D
    reducer = PCA(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(np.array(embeddings))
    
    # Store for later visualization
    st.session_state.embedding_2d = embedding_2d
    
    # Create hover texts - truncated chunk text
    hover_texts = []
    for i, chunk in enumerate(text_chunks):
        # Truncate long chunks for hover text
        if len(chunk) > 100:
            hover_text = chunk[:100] + "..."
        else:
            hover_text = chunk
        hover_texts.append(hover_text)
    
    # Store visualization data
    st.session_state.visualization_data = {
        "embedding_2d": embedding_2d,
        "hover_texts": hover_texts
    }

# Generate a plotly visualization of embeddings
def plot_embedding_visualization(highlight_indices=None):
    """Create an interactive plotly visualization of embeddings"""
    if st.session_state.visualization_data is None:
        return None
    
    embedding_2d = st.session_state.visualization_data["embedding_2d"]
    hover_texts = st.session_state.visualization_data["hover_texts"]
    
    # Create a DataFrame for plotting
    import pandas as pd
    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'text': hover_texts
    })
    
    # Create a color array, highlight selected points if provided
    colors = ['#1f77b4'] * len(df)  # Default blue color
    marker_sizes = [8] * len(df)    # Default marker size
    
    if highlight_indices is not None:
        for idx in highlight_indices:
            if 0 <= idx < len(colors):
                colors[idx] = '#d62728'  # Red for highlights
                marker_sizes[idx] = 12    # Larger markers for highlights
    
    # Create scatter plot
    fig = px.scatter(
        df, x='x', y='y',
        hover_data={'text': True, 'x': False, 'y': False},
        title="Document Chunk Visualization (Similar content appears closer together)"
    )
    
    # Update marker properties for custom colors and sizes
    fig.update_traces(
        marker=dict(
            color=colors,
            size=marker_sizes,
            line=dict(width=1, color='DarkSlateGrey')
        )
    )
    
    # Improve layout
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        height=600
    )
    
    return fig

# Modify the part of the Question Answering Tab where the retrieve_relevant_documents function is called

# Add this function to visualize the retrieval process
def visualize_retrieval_process(question, retrieved_docs, question_embedding, document_embeddings):
    """Visualize the document retrieval process"""
    st.subheader("🔍 Retrieval Process Visualization")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", len(document_embeddings))
    col2.metric("Retrieved Chunks", len(retrieved_docs))
    
    # Calculate average similarity
    avg_similarity = np.mean([doc.get("similarity", 0) for doc in retrieved_docs])
    col3.metric("Avg Similarity Score", f"{avg_similarity:.3f}")
    
    # Display similarity scores of retrieved documents
    similarity_data = {
        "Chunk ID": [doc["chunk_id"] for doc in retrieved_docs],
        "Similarity": [doc.get("similarity", 0) for doc in retrieved_docs],
        "Semantic Score": [doc.get("semantic_score", 0) for doc in retrieved_docs],
        "Keyword Score": [doc.get("keyword_score", 0) for doc in retrieved_docs],
    }
    
    # Create bar chart of similarity scores
    fig = px.bar(
        similarity_data,
        x="Chunk ID",
        y=["Semantic Score", "Keyword Score"],
        title="Retrieval Scores by Document Chunk",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Create table with retrieval details
    st.markdown("### Retrieved Chunks Details")
    
    for i, doc in enumerate(retrieved_docs):
        chunk_id = doc["chunk_id"]
        similarity = doc.get("similarity", 0)
        semantic_score = doc.get("semantic_score", 0)
        keyword_score = doc.get("keyword_score", 0)
        
        # Color code based on similarity score
        color_intensity = int(255 * min(1, similarity))
        bg_color = f"rgba(0, {color_intensity}, {255-color_intensity}, 0.2)"
        
        st.markdown(
            f"""
            <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <strong>Chunk {chunk_id}</strong> | 
            Combined Score: {similarity:.3f} | 
            Semantic: {semantic_score:.3f} | 
            Keyword: {keyword_score:.3f}
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        with st.expander(f"View Chunk {chunk_id} Content"):
            st.write(doc["text"])

# Query expansion for better retrieval
def expand_query(question):
    """Expand query with related terms to improve matching"""
    try:
        nlp = load_nlp_model()
        doc = nlp(question)
        
        # Extract key entities and noun phrases
        expanded_terms = []
        
        # Add named entities
        for ent in doc.ents:
            expanded_terms.append(ent.text)
        
        # Add noun phrases (but not if they're already added as entities)
        entity_texts = [ent.text.lower() for ent in doc.ents]
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in entity_texts:
                expanded_terms.append(chunk.text)
        
        # Create expanded query if there are new terms
        if expanded_terms:
            # Deduplicate by converting to set then back to list
            unique_terms = list(set(expanded_terms))
            # Limit to 3 most relevant terms
            top_terms = unique_terms[:3]
            expanded_query = question + " " + " ".join(top_terms)
            return expanded_query
        
        return question
    except:
        # In case of any error, return the original query
        return question

# Add these new functions to visualize the processing steps

# Function to visualize the chunking process
def visualize_chunking_process(original_text, chunks, chunk_metadata):
    """Create visual representation of how text was chunked"""
    st.subheader("📊 Text Chunking Visualization")
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Text Length", len(original_text))
    col2.metric("Number of Chunks", len(chunks))
    avg_chunk_size = sum(len(chunk) for chunk in chunks) / max(1, len(chunks))
    col3.metric("Average Chunk Size", f"{avg_chunk_size:.1f} chars")
    
    # Chunk size distribution
    chunk_sizes = [len(chunk) for chunk in chunks]
    
    # Create histogram of chunk sizes
    fig = px.histogram(
        x=chunk_sizes,
        nbins=20,
        title="Chunk Size Distribution",
        labels={"x": "Chunk Size (characters)", "y": "Count"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show chunk type breakdown
    chunk_types = {}
    for meta in chunk_metadata:
        chunk_type = meta.get("chunk_type", "unknown")
        if chunk_type in chunk_types:
            chunk_types[chunk_type] += 1
        else:
            chunk_types[chunk_type] = 1
    
    # Pie chart of chunk types
    if chunk_types:
        fig = px.pie(
            values=list(chunk_types.values()),
            names=list(chunk_types.keys()),
            title="Chunk Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show sample chunks with progressively lighter colors
    st.subheader("Sample Chunks")
    st.write("Showing first 5 chunks with different colors to visualize text splits:")
    
    for i, chunk in enumerate(chunks[:5]):
        # Calculate a color with decreasing opacity
        opacity = 0.9 - (i * 0.15)
        background_color = f"rgba(0, 100, 200, {opacity})"
        
        st.markdown(
            f"""
            <div style="background-color: {background_color}; 
                      padding: 10px; 
                      border-radius: 5px; 
                      margin-bottom: 10px;">
            <strong>Chunk {i+1}</strong> ({len(chunk)} chars)<br>
            {chunk[:200]}{"..." if len(chunk) > 200 else ""}
            </div>
            """, 
            unsafe_allow_html=True
        )

# Function to visualize the embedding process
def visualize_embedding_process(embeddings, text_chunks):
    """Visualize the embedding process with progressive updates"""
    st.subheader("🔄 Vector Embedding Process")
    
    # Create embedding metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Embeddings", len(embeddings))
    col2.metric("Vector Dimensions", len(embeddings[0]) if len(embeddings) > 0 else 0)
    
    # Calculate average vector magnitude
    avg_magnitude = np.mean([np.linalg.norm(emb) for emb in embeddings]) if len(embeddings) > 0 else 0
    col3.metric("Avg Vector Magnitude", f"{avg_magnitude:.2f}")
    
    # Show embedding similarity heatmap (for first 10 chunks)
    max_display = min(10, len(embeddings))
    if max_display > 1:
        similarity_matrix = cosine_similarity(embeddings[:max_display])
        
        # Create heatmap
        fig = px.imshow(
            similarity_matrix,
            labels=dict(x="Chunk Index", y="Chunk Index", color="Cosine Similarity"),
            x=[f"Chunk {i+1}" for i in range(max_display)],
            y=[f"Chunk {i+1}" for i in range(max_display)],
            color_continuous_scale="Viridis",
            title=f"Semantic Similarity Between First {max_display} Chunks"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sample text with its embedding visualization
        if max_display > 0:
            st.subheader("Sample Chunk with Embedding")
            st.markdown(f"**Chunk 1 Text:**\n{text_chunks[0][:200]}...")
            
            # Visualize the first few dimensions of the embedding
            dimensions_to_show = min(20, len(embeddings[0]))
            embedding_data = {
                'Dimension': list(range(1, dimensions_to_show + 1)),
                'Value': embeddings[0][:dimensions_to_show]
            }
            
            fig = px.bar(
                embedding_data, 
                x='Dimension', 
                y='Value',
                title="First 20 Dimensions of Chunk 1's Embedding Vector"
            )
            st.plotly_chart(fig, use_container_width=True)

# Function to show document structure extraction
def visualize_document_structure(document_structure):
    """Visualize the document structure extraction"""
    st.subheader("📑 Document Structure Extraction")
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Number of Sections", len(document_structure))
    
    # Calculate average section length
    if document_structure:
        avg_section_length = sum(len(section.get('content', '')) 
                              for section in document_structure.values()) / len(document_structure)
        col2.metric("Average Section Length", f"{avg_section_length:.1f} chars")
    
    # Create tree visualization of document structure
    if document_structure:
        # Prepare data for tree structure
        nodes = [{"id": "root", "label": "Document Root", "level": 0}]
        edges = []
        
        for section_num, section_info in document_structure.items():
            section_id = f"section_{section_num}"
            title = section_info.get('title', f'Section {section_num}')
            level = section_info.get('level', 1)
            
            # Truncate long titles
            display_title = (title[:30] + '...') if len(title) > 30 else title
            
            nodes.append({
                "id": section_id,
                "label": display_title,
                "level": level,
                "size": len(section_info.get('content', '')) / 1000  # Size based on content length
            })
            
            # Connect to parent (root for level 1, otherwise closest level-1 section)
            parent = "root"
            edges.append({"from": parent, "to": section_id})
        
        # Display as network graph
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"], label=node["label"], level=node.get("level", 0))
        for edge in edges:
            G.add_edge(edge["from"], edge["to"])
        
        # Create plot using matplotlib
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G, pos, 
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=10,
            arrows=True
        )
        
        # Convert to image and display in Streamlit
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf, use_column_width=True)
        
        # Display section titles and lengths
        st.markdown("### Document Sections")
        for section_num, section_info in document_structure.items():
            title = section_info.get('title', f'Section {section_num}')
            content_length = len(section_info.get('content', ''))
            
            st.markdown(
                f"""
                <div style="border-left: 4px solid #0066cc; padding-left: 10px; margin-bottom: 10px;">
                <strong>{title}</strong> ({content_length} chars)
                </div>
                """, 
                unsafe_allow_html=True
            )


# Calculate keyword-based relevance
def calculate_keyword_scores(question, text_chunks):
    """Calculate keyword-based relevance scores for each chunk"""
    # Extract keywords from question (excluding stopwords)
    # Use regex to find words at least 3 letters long
    keywords = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    
    # Load spaCy for better stopword filtering
    try:
        nlp = load_nlp_model()
        filtered_keywords = [word for word in keywords if not nlp.vocab[word].is_stop]
        if filtered_keywords:
            keywords = filtered_keywords
    except:
        # If spaCy fails, use the regex keywords
        pass
    
    # If no keywords found, return zero scores
    if not keywords:
        return [0.0] * len(text_chunks)
    
    # Calculate TF-IDF style scores
    chunk_scores = []
    for chunk in text_chunks:
        # Normalize chunk text
        chunk_text = chunk.lower()
        
        # Calculate term frequency for each keyword
        keyword_counts = {keyword: chunk_text.count(keyword) for keyword in keywords}
        
        # Compute score as weighted sum of keyword occurrences
        # Terms appearing in the query multiple times get higher weight
        keyword_weights = {k: 1 + 0.5 * question.lower().count(k) for k in keywords}
        score = sum(count * keyword_weights[keyword] for keyword, count in keyword_counts.items())
        
        # Normalize by chunk length to avoid favoring longer chunks too much
        normalized_score = score / (len(chunk.split()) + 10)  # +10 to avoid division by zero and reduce penalty for short chunks
        
        chunk_scores.append(normalized_score)
    
    # Normalize scores to [0,1] range
    max_score = max(chunk_scores) if max(chunk_scores) > 0 else 1.0
    normalized_scores = [score / max_score for score in chunk_scores]
    
    return normalized_scores

# IMPROVED: Enhanced document retrieval with OpenAI embeddings
def enhanced_retrieve_relevant_documents(question, text_chunks, embeddings, chunk_metadata, client, top_k=8, semantic_weight=0.7):
    """Enhanced retrieval with OpenAI embeddings and hybrid matching"""
    # Return early if embeddings are not available
    if embeddings is None or len(embeddings) == 0:
        st.error("No document embeddings available.")
        return [], []
    
    # Expand the query with context
    expanded_query = expand_query(question)
    
    try:
        # Get embedding for the question using OpenAI
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[expanded_query]
        )
        question_embedding = response.data[0].embedding
        
        # Calculate cosine similarity for semantic search
        question_embedding_array = np.array(question_embedding).reshape(1, -1)
        embeddings_array = np.array(embeddings)
        similarities = cosine_similarity(question_embedding_array, embeddings_array)[0]
        
        # Add keyword matching component
        keyword_scores = calculate_keyword_scores(question, text_chunks)
        
        # Combine scores with configurable weighting
        combined_scores = semantic_weight * similarities + (1 - semantic_weight) * np.array(keyword_scores)
        
        # Get top-k combined results
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        # Get the corresponding chunks and their metadata
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": idx,
                "text": text_chunks[idx],
                "metadata": chunk_metadata[idx],
                "similarity": combined_scores[idx],
                "vector": embeddings[idx] if idx < len(embeddings) else None,
                "semantic_score": similarities[idx],
                "keyword_score": keyword_scores[idx]
            })
        
        # Store the question embedding for later analysis
        st.session_state.last_question_embedding = question_embedding
        
        return results, top_indices
        
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return [], []

# Generate an answer using the RAG approach
def generate_rag_answer(question, evidence, llm_pipeline, max_length=512):
    """Generate an answer using the RAG approach"""
    if llm_pipeline is None:
        return generate_rule_based_answer(question, evidence)
    
    # Format the evidence into a context string
    context_chunks = []
    for i, doc in enumerate(evidence[:5]):  # Limit to top 5 pieces of evidence for context length
        source_type = doc["metadata"].get("source_type", "document")
        section_title = doc["metadata"].get("section_title", "")
        
        # Include source information in context
        if section_title:
            context_chunks.append(f"Evidence #{i+1} (from '{section_title}'): {doc['text']}")
        else:
            context_chunks.append(f"Evidence #{i+1} (from {source_type}): {doc['text']}")
    
    context = "\n\n".join(context_chunks)
    
    # Create a prompt for the language model
    prompt = f"""
You are a helpful AI assistant that answers questions about various types of content including text, images, and audio.

Question: {question}

Evidence from various sources:
{context}

Based on the above evidence, please provide a comprehensive and accurate answer to the question. 
Only use information from the provided context. Be specific and reference relevant content.
Keep your response focused on the user's question.

Answer:
"""
    
    # Generate the answer
    try:
        # Different handling based on model type
        if "TinyLlama" in llm_pipeline.model.config._name_or_path:
            # TinyLlama uses a chat format
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = llm_pipeline.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            outputs = llm_pipeline(
                formatted_prompt, 
                max_new_tokens=max_length,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            generated_text = outputs[0]["generated_text"]
            # Extract the model's response from the full generated text
            answer = generated_text.split("Answer:")[-1].strip()
            
        elif "dolly" in llm_pipeline.model.config._name_or_path.lower():
            # Dolly format
            outputs = llm_pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            answer = outputs[0]["generated_text"]
            
        else:
            # Generic format
            outputs = llm_pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )
            
            generated_text = outputs[0]["generated_text"]
            # Try to extract just the answer part
            answer = generated_text.replace(prompt, "").strip()
        
        # Clean up the answer if it's too verbose
        if len(answer.split()) > 300:
            answer = "\n".join(answer.split("\n")[:10]) + "\n..."
            
        return answer, evidence
        
    except Exception as e:
        st.error(f"Error generating answer with language model: {str(e)}")
        # Fall back to rule-based answer generation
        return generate_rule_based_answer(question, evidence)

def generate_rule_based_answer(question, evidence):
    """Generate an answer using rule-based approach as fallback"""
    # Start building the answer
    answer = "Based on the available content, I found this relevant information:\n\n"
    
    # Add the most relevant evidence
    if evidence:
        # Get the most relevant evidence text
        top_evidence = evidence[0]["text"]
        
        # Clean and format the evidence
        clean_evidence = re.sub(r'\s+', ' ', top_evidence).strip()
        answer += clean_evidence + "\n\n"
    
    # Add additional context statement if we have more evidence
    if len(evidence) > 1:
        answer += "Additional context from other sources supports this answer."
    
    return answer, evidence

# Process PDF with OpenAI embeddings
# Modify the process_pdf_with_openai function to include visualization
def process_pdf_with_openai(uploaded_file, client):
    """Process PDF with OpenAI embeddings and visualize the steps"""
    # Create a status container for processing steps
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    with st.spinner("Processing PDF... This may take a few minutes."):
        try:
            # Step 1: Extract text from PDF (20%)
            status_container.info("Step 1/5: Extracting text from PDF...")
            full_text, pages, document_structure = extract_text_from_pdf(uploaded_file)
            progress_bar.progress(20)
            
            if not full_text.strip():
                st.error("Could not extract text from the PDF. The document might be scanned or protected.")
                return False
            
            # Visualize document structure
            visualize_document_structure(document_structure)
            
            # Store document structure
            st.session_state.document_sections = document_structure
            
            # Step 2: Split text into chunks (40%)
            status_container.info("Step 2/5: Splitting text into semantic chunks...")
            text_chunks, chunk_metadata = improved_split_text(pages, document_structure)
            st.session_state.text_chunks = text_chunks
            st.session_state.chunk_metadata = chunk_metadata
            progress_bar.progress(40)
            
            # Visualize chunking process
            visualize_chunking_process(full_text, text_chunks, chunk_metadata)
            
            if client is None:
                st.error("OpenAI client not initialized. Please check your API key.")
                return False
            
            # Step 3: Create document embeddings (60%)
            status_container.info("Step 3/5: Creating document embeddings with OpenAI...")
            with st.spinner("Creating document embeddings with OpenAI..."):
                document_embeddings = create_document_embeddings(text_chunks, client)
                if document_embeddings is None:
                    st.error("Failed to create embeddings. Check your OpenAI API key and rate limits.")
                    return False
                    
                st.session_state.document_embeddings = document_embeddings
            progress_bar.progress(80)
            
            # Visualize embedding process
            visualize_embedding_process(document_embeddings, text_chunks)
            
            # Step 4: Load language model (100%)
            status_container.info("Step 4/5: Loading language model for answer generation...")
            if st.session_state.llm_model is None:
                st.session_state.llm_tokenizer, st.session_state.llm_model, st.session_state.llm_pipeline = load_language_model(llm_model_name)
            progress_bar.progress(90)
            
            # Step 5: Finalize processing
            status_container.info("Step 5/5: Finalizing processing...")
            progress_bar.progress(100)
            
            status_container.success(f"PDF '{uploaded_file.name}' processed successfully!")
            return True
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.exception(e)  # This will display the full traceback
            return False

# Process image and add to knowledge base
def process_image_to_knowledge_base(image, filename, processor_type="caption"):
    """Process an image and add it to the knowledge base"""
    if st.session_state.openai_client is None:
        st.error("OpenAI client not initialized. Please check your API key.")
        return False
    
    try:
        # Process image based on selected method
        if processor_type == "ocr":
            processed_text = process_image(image, "ocr")
            metadata_type = "OCR text"
            chunk_type = "image_text"
        elif processor_type == "caption":
            processed_text = process_image(image, "caption")
            metadata_type = "Caption"
            chunk_type = "image_caption"
        else:  # classification
            processed_text = process_image(image, "classification")
            metadata_type = "Classification"
            chunk_type = "image_objects"
        
        if not processed_text or processed_text.startswith("Error") or processed_text.startswith("Failed"):
            st.error(f"Failed to process image: {processed_text}")
            return False
        
        # Add image information to knowledge base
        new_chunk_id = len(st.session_state.text_chunks)
        
        # Add context about the image to the text
        enhanced_text = f"Image {metadata_type}: {processed_text}"
        st.session_state.text_chunks.append(enhanced_text)
        st.session_state.chunk_metadata.append({
            "source_type": "image",
            "file_name": filename,
            "extraction_method": metadata_type.lower(),
            "chunk_type": chunk_type
        })
        
        # Create embedding
        new_embedding = st.session_state.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[enhanced_text]
        ).data[0].embedding
        
        # Add embedding to existing embeddings
        if st.session_state.document_embeddings is not None:
            st.session_state.document_embeddings = np.vstack([st.session_state.document_embeddings, [new_embedding]])
        else:
            st.session_state.document_embeddings = np.array([new_embedding])
        
        # Store image data
        image_data = {
            "image": image,
            "filename": filename,
            "processed_text": processed_text,
            "type": metadata_type
        }
        st.session_state.images.append(image_data)
        
        # Update visualization
        if st.session_state.visualization_data is not None:
            try:
                create_embedding_visualization(st.session_state.document_embeddings, st.session_state.text_chunks)
            except:
                pass
        
        return True
    
    except Exception as e:
        st.error(f"Error adding image to knowledge base: {str(e)}")
        return False

# Create tabs for different content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Documents", 
    "🖼️ Images", 
    "🔊 Audio", 
    "🧩 Content Explorer",
    "❓ Ask Questions"
])

# Document Upload Tab
with tab1:
    st.header("Upload Documents")
    
    # Check for OpenAI client before allowing upload
    if st.session_state.openai_client is not None:
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf", key="pdf_upload")
        
        if uploaded_file:
            # Process PDF with OpenAI embeddings
            if not st.session_state.pdf_processed:
                if process_pdf_with_openai(uploaded_file, st.session_state.openai_client):
                    st.session_state.pdf_processed = True
                    
                    # Display document structure
                    if st.session_state.document_sections:
                        st.subheader("Content Found in Document")
                        for section_num, section_info in st.session_state.document_sections.items():
                            st.markdown(f"**{section_info['title']}**")
            else:
                st.success(f"PDF '{uploaded_file.name}' already processed!")
                
                # Option to process another document
                if st.button("Process Another Document"):
                    st.session_state.pdf_processed = False
                    st.experimental_rerun()
    else:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")

# Image Upload Tab
with tab2:
    st.header("Add Images")
    
    if st.session_state.openai_client is not None:
        # Image upload 
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_upload")
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            processor_selection = st.radio(
                "Image Processing Method", 
                ["Caption Image", "Extract Text (OCR)", "Classify Image Content"],
                index=0
            )
        
        # Map selection to processor type
        processor_type_map = {
            "Caption Image": "caption",
            "Extract Text (OCR)": "ocr",
            "Classify Image Content": "classification"
        }
        processor_type = processor_type_map.get(processor_selection, "caption")
        
        if image_file is not None:
            # Load and display image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process image
            with st.spinner(f"Processing image with {processor_selection}..."):
                result = process_image(image, processor_type)
                st.subheader("Processing Result:")
                st.write(result)
                
                # Add to knowledge base button
                if result and not result.startswith(("Error", "Failed")):
                    if st.button("Add to Knowledge Base"):
                        if process_image_to_knowledge_base(image, image_file.name, processor_type):
                            st.success("✅ Image added to knowledge base!")
        
        # Display existing images in knowledge base
        if st.session_state.images:
            st.header("Images in Knowledge Base")
            image_cols = st.columns(3)
            
            for i, img_data in enumerate(st.session_state.images):
                with image_cols[i % 3]:
                    st.image(img_data["image"], width=200)
                    st.caption(f"{img_data['type']}: {img_data['processed_text'][:100]}...")
    else:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")

# Audio Tab
with tab3:
    # Use the new audio tab implementation
    build_audio_tab()
    
    # Show requirements at the bottom
    with st.expander("Audio Processing Requirements"):
        display_audio_requirements()

# Content Explorer Tab
with tab4:
    st.header("Knowledge Base Explorer")
    
    if len(st.session_state.text_chunks) > 0:
        # Content statistics
        st.subheader("Content Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Text Chunks", len(st.session_state.text_chunks))
        col2.metric("Images", len(st.session_state.images))
        col3.metric("Audio Files", len(st.session_state.audio_files))
        
        # Content type breakdown
        chunk_types = {}
        for meta in st.session_state.chunk_metadata:
            chunk_type = meta.get("chunk_type", "unknown")
            source_type = meta.get("source_type", "unknown")
            key = f"{source_type}: {chunk_type}"
            
            if key in chunk_types:
                chunk_types[key] += 1
            else:
                chunk_types[key] = 1
        
        # Create pie chart for content types
        if chunk_types:
            fig = px.pie(
                values=list(chunk_types.values()),
                names=list(chunk_types.keys()),
                title="Content Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Vector space visualization
        st.subheader("Content Vector Space")
        if st.session_state.visualization_data is not None:
            fig = plot_embedding_visualization()
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Each point represents a chunk of content. Similar content appears closer together.")
        
        # Content browser
        st.subheader("Browse Content Chunks")
        
        # Filter options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            filter_source = st.selectbox(
                "Filter by Source Type",
                ["All"] + list(set(meta.get("source_type", "unknown") for meta in st.session_state.chunk_metadata))
            )
        
        with filter_col2:
            filter_type = st.selectbox(
                "Filter by Chunk Type",
                ["All"] + list(set(meta.get("chunk_type", "unknown") for meta in st.session_state.chunk_metadata))
            )
        
        # Apply filters
        filtered_indices = []
        for i, meta in enumerate(st.session_state.chunk_metadata):
            source_match = filter_source == "All" or meta.get("source_type", "unknown") == filter_source
            type_match = filter_type == "All" or meta.get("chunk_type", "unknown") == filter_type
            
            if source_match and type_match:
                filtered_indices.append(i)
        
        # Display chunks with pagination
        items_per_page = 5
        total_pages = (len(filtered_indices) + items_per_page - 1) // items_per_page
        
        if total_pages > 0:
            page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page_num - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_indices))
            
            for i in range(start_idx, end_idx):
                chunk_idx = filtered_indices[i]
                chunk = st.session_state.text_chunks[chunk_idx]
                meta = st.session_state.chunk_metadata[chunk_idx]
                
                st.markdown(f"**Chunk #{chunk_idx}**")
                st.markdown(f"*Source: {meta.get('source_type', 'unknown')} | "
                           f"Type: {meta.get('chunk_type', 'unknown')}*")
                
                # Display content
                with st.expander("View Content"):
                    # Show image if it's an image source
                    if meta.get("source_type") == "image" and meta.get("file_name"):
                        for img_data in st.session_state.images:
                            if img_data.get("filename") == meta.get("file_name"):
                                st.image(img_data["image"], width=300)
                                break
                    
                    # Show audio if it's an audio source
                    if meta.get("source_type") == "audio" and meta.get("file_name"):
                        for audio_data in st.session_state.audio_files:
                            if audio_data.get("file_name") == meta.get("file_name"):
                                if "audio_bytes" in audio_data:
                                    st.audio(audio_data["audio_bytes"])
                                break
                    
                    # Show text content
                    st.write(chunk)
                    
                    # Show embedding visualization
                    if st.session_state.document_embeddings is not None and chunk_idx < len(st.session_state.document_embeddings):
                        st.write("First 10 dimensions of the embedding vector:")
                        embedding = st.session_state.document_embeddings[chunk_idx]
                        st.line_chart(embedding[:10])
                
                st.markdown("---")
            
            # Pagination controls
            st.write(f"Page {page_num} of {total_pages}")
        else:
            st.write("No chunks match the selected filters.")
    else:
        st.info("No content has been added to the knowledge base yet. Please add some content first.")

# Question Answering Tab
with tab5:
    st.header("Ask Questions About Your Content")
    
    if len(st.session_state.text_chunks) > 0 and st.session_state.openai_client is not None:
        # Question input with examples
        st.markdown("**Examples:**")
        example_col1, example_col2 = st.columns(2)
        with example_col1:
            if st.button("📚 What happens at the end of the Amarok story?"):
                question = "What happens at the end of the Amarok story?"
                st.session_state.last_question = question
        
        with example_col2:
            if st.button("🧠 Compare the main characters in the stories"):
                question = "Compare the main characters in all the stories"
                st.session_state.last_question = question
        
        # Question input field
        if "last_question" in st.session_state:
            question = st.text_input(
                "Type your question:", 
                value=st.session_state.last_question,
                key="question_input"
            )
        else:
            question = st.text_input(
                "Type your question:", 
                placeholder="Example: What happens to Amarok at the end of the story?",
                key="question_input"
            )
        
        if question:
            # Set this as the last question for reuse
            st.session_state.last_question = question
            
            try:
                # Get retrieval settings from sidebar
                retrieval_top_k = top_k
                retrieval_semantic_weight = semantic_weight
                
                # Enhanced retrieval with OpenAI embeddings
                with st.spinner("Retrieving relevant information..."):
                    retrieved_docs, highlight_indices = enhanced_retrieve_relevant_documents(
                        question,
                        st.session_state.text_chunks,
                        st.session_state.document_embeddings,
                        st.session_state.chunk_metadata,
                        st.session_state.openai_client,
                        top_k=retrieval_top_k,
                        semantic_weight=retrieval_semantic_weight
                    )
                
                # Visualize the retrieval process
                visualize_retrieval_process(
                    question, 
                    retrieved_docs, 
                    st.session_state.last_question_embedding,
                    st.session_state.document_embeddings
                )

                # Generate the answer using RAG method
                with st.spinner("Generating answer..."):
                    answer, evidence = generate_rag_answer(
                        question,
                        retrieved_docs,
                        st.session_state.llm_pipeline
                    )
                
                # Display the answer
                st.markdown("### Answer")
                st.write(answer)
                
                # User feedback
                st.write("Was this answer helpful?")
                col1, col2 = st.columns(2)
                if col1.button("👍 Yes"):
                    st.success("Thank you for your feedback!")
                if col2.button("👎 No"):
                    st.warning("You can try adjusting the retrieval settings in the sidebar for better results.")
                
                # Show visualization with highlighted chunks
                st.subheader("Retrieved Content Visualization")
                if st.session_state.visualization_data is not None:
                    fig = plot_embedding_visualization(highlight_indices)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Red points show the chunks used to answer your question.")
                
                # Display the sources with better organization
                st.subheader("Source Evidence")
                
                # Create tabs for different source types
                source_types = set(e["metadata"].get("source_type", "other") for e in evidence)
                if len(source_types) > 1:
                    source_tabs = st.tabs([s.capitalize() for s in source_types])
                    
                    for i, (source_type, tab) in enumerate(zip(source_types, source_tabs)):
                        with tab:
                            # Filter evidence by source type
                            type_evidence = [e for e in evidence if e["metadata"].get("source_type", "other") == source_type]
                            
                            for j, e in enumerate(type_evidence):
                                st.markdown(f"**Source {j+1}**")
                                
                                # Show image if it's an image source
                                if source_type == "image" and e["metadata"].get("file_name"):
                                    for img_data in st.session_state.images:
                                        if img_data.get("filename") == e["metadata"].get("file_name"):
                                            st.image(img_data["image"], width=300)
                                            break
                                
                                # Show audio if it's an audio source
                                if source_type == "audio" and e["metadata"].get("file_name"):
                                    for audio_data in st.session_state.audio_files:
                                        if audio_data.get("file_name") == e["metadata"].get("file_name"):
                                            if "audio_bytes" in audio_data:
                                                st.audio(audio_data["audio_bytes"])
                                            break
                                
                                # Show text content
                                st.markdown(e["text"])
                                
                                # Show metadata
                                with st.expander("View Source Metadata"):
                                    st.json(e["metadata"])
                                
                                st.markdown("---")
                else:
                    # If only one source type, show directly
                    for i, e in enumerate(evidence):
                        st.markdown(f"**Source {i+1}**")
                        source_type = e["metadata"].get("source_type", "unknown")
                        
                        # Show image if it's an image source
                        if source_type == "image" and e["metadata"].get("file_name"):
                            for img_data in st.session_state.images:
                                if img_data.get("filename") == e["metadata"].get("file_name"):
                                    st.image(img_data["image"], width=300)
                                    break
                        
                        # Show audio if it's an audio source
                        if source_type == "audio" and e["metadata"].get("file_name"):
                            for audio_data in st.session_state.audio_files:
                                if audio_data.get("file_name") == e["metadata"].get("file_name"):
                                    if "audio_bytes" in audio_data:
                                        st.audio(audio_data["audio_bytes"])
                                    break
                        
                        st.markdown(e["text"])
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.exception(e)  # This will display the full traceback
    else:
        st.info("Please add some content to the knowledge base first, and make sure your OpenAI API key is set.")

# Footer
st.markdown("---")
st.markdown("Multimodal RAG System: Knowledge-Enhanced Question Answering Across Multiple Content Types")
st.markdown("Developer by Jugal Wable")