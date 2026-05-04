import gradio as gr
from core.chat_interface import ChatInterface
from core.document_manager import DocumentManager
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import OpenAIEmbeddings

def create_gradio_ui(rag_system):
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)
    
    def format_file_list():
        files = doc_manager.get_markdown_files()
        if not files:
            return "📭 No documents available in the knowledge base"
        return "\n".join([f"{f}" for f in files])
    
    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, format_file_list()
            
        added, skipped = doc_manager.add_documents(
            files, 
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        gr.Info(f"✅ Added: {added} | Skipped: {skipped}")
        return None, format_file_list()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info(f"🗑️ Removed all documents")
        return format_file_list()
    
    # --- SIMPLIFIED CHAT HANDLER ---
    def chat_handler(msg, hist):
        for chunk_text in chat_interface.chat(msg, hist):
            yield chunk_text
    # -------------------------------
    
    def clear_chat_handler():
        chat_interface.clear_session()
        rag_system.reset_thread() 

    def run_ragas_ui(query, draft, context):
        try:
            import numpy as np
            # We import the exact same CrossEncoder model we already loaded in nodes.py
            # so we don't waste memory loading it twice!
            from rag_agent.nodes import delta_model
            
            if not context.strip() or not draft.strip():
                return "❌ Please provide both the Retrieved Context and the Draft Response."
            
            # Run the mathematical comparison
            raw_scores = delta_model.predict([(context.strip(), draft.strip())])
            
            # Convert to probabilities (0.0 to 1.0)
            probabilities = np.exp(raw_scores[0]) / np.sum(np.exp(raw_scores[0]))
            contradiction_prob = probabilities[0]
            entailment_prob = probabilities[1]
            neutral_prob = probabilities[2]
            
            # Determine Pass/Fail based on the 0.5 threshold we set in nodes.py
            if contradiction_prob > 0.5:
                status = "❌ FAIL (Guardrail would block this)"
            else:
                status = "✅ PASS (Guardrail would allow this)"
            
            # Format the output beautifully
            result_text = (
                f"**System Status:** {status}\n\n"
                f"**Obligation Delta (Contradiction):** {contradiction_prob:.4f}\n"
                f"*(If this is > 0.5, it means the draft stripped away or contradicted the context)*\n\n"
                f"**Entailment (Factual Match):** {entailment_prob:.4f}\n"
                f"**Neutral (Unrelated Info):** {neutral_prob:.4f}"
            )
            
            return result_text
            
        except Exception as e:
            return f"❌ Evaluation Failed: {str(e)}"
            
        except Exception as e:
            return f"❌ Evaluation Failed: {str(e)}\n\nTip: Your CPU is timing out. Try using a context snippet of only 1-2 sentences to test."
    
    with gr.Blocks(title="Agentic RAG") as demo:
        with gr.Tab("Documents", elem_id="doc-management-tab"):
            gr.Markdown("## Add New Documents")
            gr.Markdown("Upload PDF or Markdown files. Duplicates will be automatically skipped.")
            
            files_input = gr.File(
                label="Drop PDF or Markdown files here",
                file_count="multiple",
                type="filepath",
                height=200,
                show_label=False
            )
            
            add_btn = gr.Button("Add Documents", variant="primary", size="md")
            
            gr.Markdown("## Current Documents in the Knowledge Base")
            file_list = gr.Textbox(
                value=format_file_list(),
                interactive=False,
                lines=7,
                max_lines=10,
                elem_id="file-list-box",
                show_label=False
            )
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="md")
                clear_btn = gr.Button("Clear All", variant="stop", size="md")
            
            add_btn.click(
                upload_handler, 
                [files_input], 
                [files_input, file_list], 
                show_progress="corner"
            )
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_handler, None, file_list)
        
        with gr.Tab("Chat"):
            # Define the visual look of the chatbot
            my_chatbot = gr.Chatbot(
                height=600, 
                placeholder="Ask me anything about your documents!",
                show_label=False
            )
            
            # ChatInterface automatically builds the input bar, submit button, and clear button!
            gr.ChatInterface(
                fn=chat_handler, 
                chatbot=my_chatbot
            )

        with gr.Tab("Ragas Evaluation"):
            gr.Markdown("## Run Mathematical Accuracy Assessment")
            gr.Markdown("Warning: This requires multiple LLM inference passes and may take 15-20 minutes on local hardware.")
            
            ragas_q = gr.Textbox(label="User Query")
            ragas_d = gr.Textbox(label="Draft Response")
            ragas_c = gr.Textbox(label="Retrieved Context (Paste the raw text the model used)")
            
            ragas_btn = gr.Button("Calculate Score", variant="primary")
            ragas_out = gr.Textbox(label="Ragas Score output", lines=8)
            
            ragas_btn.click(run_ragas_ui, inputs=[ragas_q, ragas_d, ragas_c], outputs=[ragas_out])
            
    return demo