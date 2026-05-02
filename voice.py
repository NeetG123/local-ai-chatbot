"""
Voice I/O module.

SpeechToText  — Whisper-based transcription (file or live mic)
TextToSpeech  — pyttsx3-based playback (Windows SAPI5 / macOS / Linux espeak)

Coqui TTS alternative is documented at the bottom of the file.
"""

import os
import threading

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import whisper
import pyttsx3

import config


class SpeechToText:
    def __init__(self):
        print(f"[INFO] Loading Whisper '{config.WHISPER_MODEL}' model …")
        self.model = whisper.load_model(config.WHISPER_MODEL)
        os.makedirs(config.AUDIO_TEMP_DIR, exist_ok=True)

    def transcribe_file(self, filepath: str) -> str:
        """Transcribe an audio file (MP3, WAV, M4A, …).

        This is the entry point used by the Gradio audio widget, which
        saves the recorded clip to a temp file before calling this method.
        """
        result = self.model.transcribe(filepath, language="en", fp16=False)
        text   = result["text"].strip()
        print(f"[STT] {text}")
        return text

    def record_and_transcribe(self) -> str:
        """Record from the default microphone and return transcribed text.

        Uses sounddevice for cross-platform mic access.
        Audio is saved to a temp WAV file because Whisper requires a path.
        """
        print(f"[REC] Recording {config.AUDIO_RECORD_SECONDS}s …")
        audio = sd.rec(
            int(config.AUDIO_RECORD_SECONDS * config.AUDIO_SAMPLE_RATE),
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[REC] Done.")

        tmp = os.path.join(config.AUDIO_TEMP_DIR, "recording.wav")
        wav.write(tmp, config.AUDIO_SAMPLE_RATE, (audio * 32767).astype(np.int16))
        return self.transcribe_file(tmp)


class TextToSpeech:
    """pyttsx3 wrapper.

    pyttsx3 is not thread-safe on Windows SAPI5 — a fresh engine instance
    is created per call and explicitly stopped to avoid COM object leaks.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def _engine(self) -> pyttsx3.Engine:
        engine = pyttsx3.init()
        engine.setProperty("rate",   config.TTS_RATE)
        engine.setProperty("volume", config.TTS_VOLUME)
        return engine

    def speak(self, text: str) -> None:
        """Speak synchronously (blocks until speech is finished)."""
        with self._lock:
            engine = self._engine()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

    def speak_async(self, text: str) -> None:
        """Speak in a background daemon thread so the UI stays responsive."""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()


# ── Coqui TTS (optional, higher-quality voices) ───────────────────
#
# Installation (do this in a fresh venv, after torch is already installed):
#   pip install TTS==0.22.0
#
# Windows note: Coqui requires Microsoft C++ Build Tools and a matching
# torch ABI. Expect DLL errors if torch was upgraded after install.
#
# Usage example:
#
#   from TTS.api import TTS as CoquiTTS
#
#   class CoquiTextToSpeech:
#       def __init__(self):
#           self.tts = CoquiTTS("tts_models/en/ljspeech/tacotron2-DDC")
#
#       def speak(self, text: str, output_path: str = "audio_temp/out.wav"):
#           self.tts.tts_to_file(text=text, file_path=output_path)
#           # play output_path with sounddevice or os.system("start out.wav")
