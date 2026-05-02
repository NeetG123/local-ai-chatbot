"""
Gradio web UI — entry point.

Run:  python app.py
Open: http://localhost:7860
"""

import shutil
from pathlib import Path

import gradio as gr

import config
from ingest import run_ingestion
from rag_chain import RAGChatbot
from voice import SpeechToText, TextToSpeech

# ── Singletons (loaded once at startup) ──────────────────────────
chatbot = RAGChatbot()
stt     = SpeechToText()
tts     = TextToSpeech()


# ── Helpers ───────────────────────────────────────────────────────

def _format_sources(sources: list[str]) -> str:
    if not sources:
        return "General knowledge (no documents matched)"
    return "\n".join(f"• {Path(s).name}" for s in sources)


# ── Event handlers ────────────────────────────────────────────────

def handle_text(user_msg: str, history: list, speak: bool):
    if not user_msg.strip():
        return history, "", ""
    answer, sources = chatbot.query(user_msg)
    history.append((user_msg, answer))
    if speak:
        tts.speak_async(answer)
    return history, "", _format_sources(sources)


def handle_voice(audio_path: str | None, history: list, speak: bool):
    if audio_path is None:
        return history, "No audio received.", ""
    user_msg        = stt.transcribe_file(audio_path)
    answer, sources = chatbot.query(user_msg)
    history.append((f"🎤 {user_msg}", answer))
    if speak:
        tts.speak_async(answer)
    return history, f'You said: "{user_msg}"', _format_sources(sources)


def handle_upload(files) -> str:
    if not files:
        return "No files selected."

    docs_path = Path(config.DOCS_DIR)
    docs_path.mkdir(exist_ok=True)
    saved = []
    for f in files:
        dest = docs_path / Path(f.name).name
        shutil.copy(f.name, dest)
        saved.append(dest.name)

    run_ingestion()
    chatbot.refresh_retriever()
    return f"Ingested {len(saved)} file(s): {', '.join(saved)}"


def handle_clear():
    chatbot.clear_history()
    return [], "", ""


# ── UI layout ─────────────────────────────────────────────────────

with gr.Blocks(title="Local AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Local AI Chatbot\n"
        "Powered by **LM Studio** · **LangChain** · **ChromaDB** · **Whisper** — fully offline."
    )

    with gr.Row():

        # ── Left: conversation ────────────────────────────────────
        with gr.Column(scale=3):
            chatbox = gr.Chatbot(label="Conversation", height=460, bubble_full_width=False)

            with gr.Row():
                text_in  = gr.Textbox(
                    placeholder="Ask anything… (Enter to send)",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            speak_toggle = gr.Checkbox(label="Speak responses aloud (TTS)", value=False)

        # ── Right: voice + docs + session ────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### Voice Input")
            audio_in  = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload audio",
            )
            voice_btn = gr.Button("Transcribe & Ask", variant="secondary")
            voice_txt = gr.Textbox(label="Transcription", interactive=False, lines=2)

            gr.Markdown("### Documents")
            file_in      = gr.File(
                label="Upload PDF / DOCX / TXT",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"],
            )
            ingest_btn    = gr.Button("Ingest Documents", variant="secondary")
            ingest_status = gr.Textbox(label="Status", interactive=False, lines=2)

            gr.Markdown("### Session")
            clear_btn = gr.Button("Clear Chat History", variant="stop")

    source_box = gr.Textbox(label="Sources used in last answer", interactive=False, lines=3)

    # ── Wire events ───────────────────────────────────────────────
    send_btn.click(
        handle_text,
        inputs=[text_in, chatbox, speak_toggle],
        outputs=[chatbox, text_in, source_box],
    )
    text_in.submit(
        handle_text,
        inputs=[text_in, chatbox, speak_toggle],
        outputs=[chatbox, text_in, source_box],
    )
    voice_btn.click(
        handle_voice,
        inputs=[audio_in, chatbox, speak_toggle],
        outputs=[chatbox, voice_txt, source_box],
    )
    ingest_btn.click(
        handle_upload,
        inputs=[file_in],
        outputs=[ingest_status],
    )
    clear_btn.click(
        handle_clear,
        outputs=[chatbox, text_in, source_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
