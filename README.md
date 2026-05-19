# ⚖️ Local Agentic RAG for Indian Law: Hallucination-Free Legal AI

A highly optimized, multi-agent Retrieval-Augmented Generation (RAG) pipeline built to navigate the complexities of Indian Legal Statutes (IPC, CrPC, MVA, CPC, IDA). 

Designed to run **entirely locally on consumer-grade hardware** (tested on Intel Iris Xe integrated GPUs), this project tackles the two biggest challenges in Legal AI: **LLM Hallucinations** and **Hardware VRAM Constraints**.

## 🚀 Key Features

* **Multi-Agent Orchestration (LangGraph):** Replaces the traditional "single-shot" RAG approach with a dynamic graph. Specialized nodes handle query rewriting, autonomous research, and final answer aggregation.
* **Mathematical Factual Guardrails (DeBERTa NLI):** Integrates a Natural Language Inference CrossEncoder to mathematically verify the LLM's draft against the raw retrieved legal text. If the "Obligation Delta" (Contradiction score) exceeds the threshold, the system flags the hallucination.
* **Sequential Tool Calling:** Prevents Vulkan Out-Of-Memory (OOM) crashes by forcing the agent to execute bite-sized, sequential database searches (multi-hop reasoning) rather than dumping massive context blocks into the LLM at once.
* **Parent-Child Chunking:** Maintains the structural integrity of dense legal texts in the vector database. Searches match against granular "child" chunks but return the full "parent" section to ensure critical provisos and exceptions are never missed.
* **"Glass Box" Streaming UI:** A custom Gradio interface that transparently streams the agent's live tool calls and thought processes while silencing intermediate drafts for a clean user experience.

## 🛠️ Tech Stack

* **LLM Engine:** Qwen 2.5 (7B) Instruct (via LM Studio)
* **Orchestration:** LangGraph / LangChain
* **Vector Database:** Qdrant (Local)
* **Embeddings:** `Gemini embeddings`
* **Guardrail Model:** DeBERTa (CrossEncoder)
* **Frontend:** Gradio

## 🧠 Why this Architecture?

Standard RAG systems often fail in legal domains because they retrieve disconnected paragraphs and suffer from "Lost in the Middle" syndrome, leading to dangerous legal hallucinations. Furthermore, loading 10+ chunks of text into a local model typically crashes consumer GPUs. 

This project solves this by giving the LLM **agency**. The Qwen 2.5 research agent searches the database, reads a small safe limit of chunks, realizes if it is missing procedural rules or exceptions, and autonomously fires a *second* search to gather the rest of the puzzle before generating a verified answer.

## 📊 Performance on Legal Edge Cases
This system has been successfully stress-tested against brutal legal retrieval traps, including:
- **Cross-Statute Multi-Hop Reasoning:** e.g., Finding an offence in the IPC, but fetching the reporting procedure from the CrPC.
- **Mathematical Penalty Synthesis:** Adding overlapping fines and jail terms across different sections of the Motor Vehicles Act (MVA).
- **Nested Proviso Traps:** Identifying absolute exceptions hidden deep within the Code of Civil Procedure (CPC).
