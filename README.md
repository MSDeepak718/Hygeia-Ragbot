# 🤖 Hygeia - A Medical RAG Assistant

A full-stack AI-powered chatbot application that allows users to ask medical-related questions. It uses **LangChain**, **Ollama**, and **FAISS** on the backend to implement a Retrieval-Augmented Generation (RAG) pipeline, and a **React + TypeScript** frontend for a clean chat interface.

---

## Features

-  **Conversational RAG**: Retains context across messages
-  **FAISS Vector Store**: Automatically created on first run if not found
-  **Local LLM & Embeddings**: Powered by Ollama models (`llama3`, `mxbai-embed-large`)
-  **FastAPI Backend**: Clean API to interface with React
-  **React Frontend**: Chat UI to interact with the assistant

---

## Prerequisites

### Backend

- Python 3.13+
- [Ollama](https://ollama.com) (installed and running locally)

> Make sure the following models are pulled:
```bash
ollama run llama3
ollama run mxbai-embed-large
```

### Frontend
- Node.js 18+
- npm

## Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/MSDeepak718/Hygeia-Ragbot.git
cd Hygeia-Ragbot
```

### 2️⃣ Setup Backend

```bash
cd backend
python -m venv env
./env/Scripts/activate  # or env\bin\activate on MacOS

pip install -r requirements.txt
```

### 3️⃣ Run FastAPI Server
```bash
uvicorn main:app --reload
```

### 4️⃣ Setup Frontend
```bash
npm install
npm run dev
```






