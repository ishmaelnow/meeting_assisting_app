from transcription import transcribe_audio_path
from summarization import summarize_meeting_with_mistral
from vector_store import build_vector_store_from_transcript, build_retriever
from qa import answer_question_with_mistral


def main():
    # 1. Pick your test audio file inside this project folder
    audio_path = "min_meeting.mp3"  # or "meeting.mp3" if you prefer

    print(f"[main] Transcribing audio: {audio_path}")
    transcript = transcribe_audio_path(audio_path)

    print("\n=== Transcript (first 500 characters) ===")
    print(transcript[:500])
    print("\n... (transcript truncated)\n")

    # 2. Summarize the full transcript with Mistral
    print("[main] Summarizing transcript with Mistral...")
    summary = summarize_meeting_with_mistral(transcript)

    print("\n=== Meeting Summary ===")
    print(summary)

    # 3. Build vector store + retriever from the transcript
    print("\n[main] Building vector store from transcript...")
    vector_store, embeddings = build_vector_store_from_transcript(transcript)
    retriever = build_retriever(vector_store, k=4)
    print("[main] Vector store and retriever ready.")

    # 4. Ask a couple of example questions about the meeting
    example_questions = [
        "What was the main topic of this discussion?",
        "What did they say about vector indices or retrievers?",
    ]

    for q in example_questions:
        print(f"\n[Q&A] Question: {q}")
        answer = answer_question_with_mistral(q, retriever)
        print("[Q&A] Answer:", answer)


if __name__ == "__main__":
    main()
