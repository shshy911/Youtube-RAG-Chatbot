# 🎥 YouTube RAG Chatbot

A conversational AI chatbot that lets users chat with the contents of a YouTube video.

The application extracts a video's transcript, converts it into embeddings using Sentence Transformers, stores them in ChromaDB, and uses Retrieval-Augmented Generation (RAG) to answer questions grounded in the video's content. It also supports multi-turn conversations through chat history, allowing users to ask follow-up questions naturally.

## ✨ Features

* Extract transcripts from YouTube videos
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Context-aware follow-up conversations
* FastAPI backend
* Simple and responsive frontend

## 🛠️ Tech Stack

* FastAPI
* LangChain
* ChromaDB
* Hugging Face
* Sentence Transformers
* HTML, CSS, JavaScript

## 🚀 Setup

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create a `.env` file

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

3. Run the backend

```bash
uvicorn backend:app --reload
```

4. Open `frontend.html` and start chatting with YouTube videos.
