# Multimodal RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that enables question answering across multiple modalities including text documents, images, and audio content.

## Project Overview

This system implements a comprehensive knowledge base that ingests content from various sources, processes it, and enables natural language queries with contextually relevant answers. The application uses state-of-the-art AI technologies to understand and retrieve information from multimodal content.

## Features

- **Document Processing**: Extract and chunk text from PDF documents with semantic boundary awareness
- **Image Understanding**: Process images using captioning, OCR, and classification
- **Audio Processing**: Transcribe spoken content using OpenAI Whisper
- **Advanced Retrieval**: Hybrid retrieval combining semantic similarity and keyword matching
- **Knowledge Visualization**: Interactive visualization of content embeddings
- **Content Explorer**: Browse, filter, and examine your knowledge base
- **Customizable Question Answering**: Configure retrieval parameters for optimal results

## Technology Stack

### Core Components

- **Python**: Primary programming language
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **OpenAI API**: Embeddings and audio transcription
- **Hugging Face Transformers**: Language and vision models

### Retrieval-Augmented Generation (RAG)

- OpenAI embeddings for semantic search
- Hybrid retrieval combining vector similarity and keyword matching
- Custom chunking strategies for narrative text
- PCA-based content visualization

### Multimodal Integration

- **Text**: PDF processing with PyPDF2
- **Images**: Vision models for captioning, OCR, and classification
- **Audio**: Whisper API for high-quality transcription

### LLM Integration

- TinyLlama, OPT, and Dolly models for answer generation
- Context-aware prompt templates
- Fallback strategies for reliability

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-rag-system.git
cd multimodal-rag-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

## Usage

1. Set your OpenAI API key in the sidebar (or in the .env file)
2. Upload PDFs, images, or record/upload audio content
3. Process content to build your knowledge base
4. Ask questions and get comprehensive answers
5. Explore visualizations and content

```bash
# Run the application
streamlit run app.py
```

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── app_new.py              # Updated application with enhancements
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (create this file for API keys)
├── .gitignore              # Git ignore file
└── README.md               # Project documentation
```

## Environment Variables

Create a `.env` file in the project root with the following:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Future Improvements

- Integration with vector databases for larger knowledge bases
- Support for more document formats
- Fine-tuning options for domain-specific adaptation
- Improved multimodal reasoning capabilities

## License

MIT

## Acknowledgments

- Hugging Face for open-source models
- OpenAI for embeddings and Whisper API
- Streamlit for the interactive web framework
