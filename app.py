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
