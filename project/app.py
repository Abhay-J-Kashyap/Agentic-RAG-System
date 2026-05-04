import sys
import os
import gradio as gr
from core.rag_system import RAGSystem
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    print("\nCreating RAG Assistant...")
    rag_instance = RAGSystem()
    rag_instance.initialize()
    demo = create_gradio_ui(rag_instance)
    print("\nLaunching RAG Assistant...")
    demo.launch(css=custom_css)