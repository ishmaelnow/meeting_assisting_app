import gradio as gr

from transcription import transcribe_audio_path
from summarization import summarize_meeting_with_mistral
from vector_store import build_vector_store_from_transcript, build_retriever
from qa import answer_question_with_mistral


# =========================================================
# Core functions used by Gradio
# =========================================================

def process_audio(audio_path: str):
    """
    Gradio callback:
    - Takes a path to an uploaded audio file
    - Transcribes it with Whisper
    - Summarizes with Mistral
    - Builds a FAISS vector store + retriever

    Returns:
    - transcript_preview (str)
    - summary_markdown (str)
    - retriever (stored in gr.State for later Q&A)
    """
    if not audio_path:
        return "No audio file uploaded yet.", "", None

    # 1. Transcribe
    try:
        transcript = transcribe_audio_path(audio_path)
    except Exception as e:
        return f"Error during transcription: {e}", "", None

    # 2. Summarize
    try:
        summary = summarize_meeting_with_mistral(transcript)
    except Exception as e:
        summary = f"Error during summarization: {e}"

    # 3. Build vector store + retriever
    try:
        vector_store, embeddings = build_vector_store_from_transcript(transcript)
        retriever = build_retriever(vector_store, k=4)
    except Exception as e:
        # We still show the transcript preview even if retrieval fails
        preview = transcript[:1500]
        error_msg = f"Error while building vector store / retriever: {e}"
        return preview, error_msg, None

    # 4. Prepare transcript preview for UI
    transcript_preview = transcript[:1500]
    if len(transcript) > 1500:
        transcript_preview += "\n\n...[truncated]"

    return transcript_preview, summary, retriever


def answer_question_ui(question: str, retriever):
    """
    Gradio callback:
    - Takes a question and the retriever stored in gr.State
    - Uses Mistral to answer based on retrieved transcript chunks
    """
    if retriever is None:
        return "Please run **Transcribe & Summarize** first so I can build the meeting memory."

    if not question or not question.strip():
        return "Please enter a question about the meeting."

    try:
        answer = answer_question_with_mistral(question, retriever)
    except Exception as e:
        answer = f"Error while answering your question: {e}"

    return answer


# =========================================================
# Gradio UI definition
# =========================================================

def build_interface():
    with gr.Blocks(title="Meeting Assistant") as demo:
        gr.Markdown(
            """
            # 🎧 Meeting Assistant

            Upload a meeting recording, then:
            1. **Transcribe & summarize** it  
            2. **Ask questions** about what was discussed using RAG (FAISS + Mistral)
            """
        )

        with gr.Row():
            # Left column: audio upload + button
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload meeting audio",
                    sources=["upload"],
                    type="filepath",   # gives us the file path as a string
                )

                transcribe_button = gr.Button("🚀 Transcribe & Summarize", variant="primary")

                # Hidden state to keep the retriever between button clicks
                retriever_state = gr.State(value=None)

            # Right column: transcript + summary
            with gr.Column(scale=2):
                transcript_box = gr.Textbox(
                    label="Transcript (preview)",
                    lines=12,
                    interactive=False,
                    placeholder="Transcript will appear here after processing.",
                )

                summary_box = gr.Markdown(
                    value="",
                    label="Meeting Summary",
                )

        gr.Markdown("## ❓ Ask questions about this meeting")

        with gr.Row():
            question_box = gr.Textbox(
                label="Your question",
                placeholder="Example: What were the main action items?",
                lines=2,
            )
            ask_button = gr.Button("💬 Ask")

        answer_box = gr.Markdown(
            label="Answer",
            value="I’ll answer your question here once the meeting has been processed.",
        )

        # Wire buttons to functions
        transcribe_button.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=[transcript_box, summary_box, retriever_state],
        )

        ask_button.click(
            fn=answer_question_ui,
            inputs=[question_box, retriever_state],
            outputs=answer_box,
        )

        return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
