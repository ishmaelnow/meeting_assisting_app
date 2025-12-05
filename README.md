---
title: Meeting Assistant – Whisper + Mistral
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Meeting Assistant – Whisper + Mistral

This app lets you:

1. Upload a meeting audio file (`.mp3`, `.wav`, etc.)
2. Transcribe it using OpenAI Whisper
3. Summarize the meeting with Mistral
4. Ask questions about the meeting using a FAISS vector store + RAG-style retrieval

## How it works

- **Transcription:** `transcription.py` uses the OpenAI API to generate a transcript.  
- **Summarization:** `summarization.py` calls Mistral to produce a structured summary.  
- **Vector Store:** `vector_store.py` builds a FAISS index over transcript chunks.  
- **Q&A:** `qa.py` retrieves relevant chunks and asks Mistral to answer questions.  
- **UI:** `gradio_app.py` defines the Gradio interface; `app.py` exposes it to Hugging Face Spaces.

### Environment variables

These are set in the Space under **Settings → Variables & secrets**:

- `OPENAI_API_KEY`
- `MISTRAL_API_KEY`
