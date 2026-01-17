# 1. INSTALL DEPENDENCIES
!pip install -q -U llama-index llama-index-llms-google-genai llama-index-embeddings-google-genai pypdf nest_asyncio gradio

import os
import nest_asyncio
import gradio as gr
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# --- INITIALIZATION ---
nest_asyncio.apply()
os.environ["GOOGLE_API_KEY"] = "ENTER YOUR API KEY" #enter your api key#

# 2026 Stable Models
Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/text-embedding-004")
Settings.chunk_size = 1024

# Global variable to hold our index
index = None

# --- BACKEND FUNCTIONS ---
def process_files(file_objs):
    """Data Engineering: Upload, Chunk, Embed, and Index"""
    global index
    if not file_objs:
        return "⚠️ Please upload at least one PDF."

    data_path = "./data_gradio"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    file_names = []
    for file in file_objs:
        name = os.path.basename(file.name)
        with open(os.path.join(data_path, name), "wb") as f:
            f.write(open(file.name, "rb").read())
        file_names.append(name)

    try:
        documents = SimpleDirectoryReader(data_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        return f"✅ Successfully indexed {len(file_names)} files: {', '.join(file_names)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_notes():
    """Generates a summary of notes from the entire index."""
    global index
    if index is None:
        return "⚠️ Please upload and 'Process' documents first."

    # We use a specific query engine for summarization
    query_engine = index.as_query_engine(response_mode="tree_summarize")

    prompt = """
    You are an expert student assistant.
    Create comprehensive, well-structured study notes from the entire provided document content.
    Use main headings, subheadings, and detailed bullet points to organize the key information logically.
    Focus on definitions, important concepts, and summaries.
    """

    try:
        # Show a loading indicator in the UI
        gr.Info("Generating study notes... This may take a minute.")
        response = query_engine.query(prompt)
        return response.response
    except Exception as e:
        return f"❌ Error generating notes: {str(e)}"
