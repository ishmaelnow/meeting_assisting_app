from transcription import transcribe_audio_path
from summarization import summarize_meeting_with_mistral


def main():
    # ✅ we now use the audio file that lives in this project folder
    audio_path = "min_meeting.mp3"

    print(f"[main] Transcribing audio: {audio_path}")
    transcript = transcribe_audio_path(audio_path)

    print("\n=== Transcript (first 500 characters) ===")
    print(transcript[:500])
    print("\n... (transcript truncated)\n")

    print("[main] Summarizing transcript with Mistral...")
    summary = summarize_meeting_with_mistral(transcript)

    print("\n=== Meeting Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
