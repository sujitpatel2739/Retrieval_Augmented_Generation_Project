# 📚 Retrieval-Augmented Generation (RAG) Assistant

A private, intelligent research assistant powered by **Retrieval-Augmented Generation (RAG)**. This system allows users to upload PDFs, index documents using vector embeddings, and ask natural language questions to receive contextually accurate answers with source references.

## 🔍 Features

- 📄 **Custom Document Uploads** (PDF, TXT, etc.)
- 🧠 **Semantic Search using Pinecone DB**
- 🤖 **Natural Language Q&A** via OpenAI / LLaMA / Mistral
- 🧾 **Cited Answers with Source Mapping**

## 🛠️ Tech Stack

- `Python 3.10+`
- `LangChain`
- `Pinecone DB`
- `OpenAI GPT-4` or `Local LLMs via Ollama`
- `PyMuPDF / pdfminer.six`
- `Streamlit` for UI
- `tiktoken` / `SentenceTransformers` for embedding

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
