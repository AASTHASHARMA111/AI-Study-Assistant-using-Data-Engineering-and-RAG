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

def chat_response(message, history):
    """RAG Retrieval & Generation with Source Attribution"""
    global index
    if index is None:
        return "⚠️ Please upload and 'Process' your documents in the sidebar first."

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(message)

    answer = f"{response.response}\n\n"
    answer += "🔍 **Sources Found:**\n"
    for node in response.source_nodes:
        fname = node.metadata.get('file_name', 'Unknown')
        page = node.metadata.get('page_label', 'Unknown')
        answer += f"- *{fname} (Page {page})* | Relevance: {node.score:.2f}\n"

    return answer

# --- CUSTOM THEME ---
# Defining a custom theme with purple and pink accents on a white background
theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="pink",
    neutral_hue="slate",
).set(
    body_background_fill="#ffffff", # Pure white background
    block_background_fill="#fdf7ff", # Very light pinkish-white for blocks
    block_border_color="#e0c3fc",   # Light purple border
    button_primary_background_fill="#9333ea", # Vibrant purple button
    button_primary_background_fill_hover="#ec4899", # Pink on hover
    button_primary_text_color="white",
    block_title_text_color="#9333ea" # Purple title color
)

# --- GRADIO DASHBOARD UI ---
with gr.Blocks(theme=theme, title="Gemini 2.5 RAG Study Assistant") as demo:
    gr.Markdown("# Data Engg + RAG Study Assistant")
    gr.Markdown("Upload your **Syllabus, PYQs, and Books** to generate notes or chat with Gemini 2.5 Flash.")

    with gr.Row():
        # Sidebar/Left Column: Data Engineering & Notes
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Data Pipeline")
            file_output = gr.File(file_count="multiple", label="Upload PDF Documents", file_types=[".pdf"])
            process_btn = gr.Button("🚀 Process & Index Data", variant="primary")
            status_msg = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---") # Separator
            gr.Markdown("### 📝 Notes Generator")
            gr.Markdown("Click below to generate comprehensive study notes from all uploaded files.")
            notes_btn = gr.Button("✨ Generate Study Notes ✨", variant="primary")
            # Use Markdown component to display formatted notes
            notes_output = gr.Markdown(label="Generated Notes")

        # Main Column: Chat Interface
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with your Knowledge Base")
            chatbot = gr.ChatInterface(
                fn=chat_response,
                examples=["Is this topic in my syllabus?", "Summarize the key concepts from Chapter 2", "Find PYQ questions related to OS Scheduling"],
                cache_examples=False,
            )
