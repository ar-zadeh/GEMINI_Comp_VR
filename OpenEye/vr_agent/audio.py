"""
vr_agent/audio.py
-----------------
AudioAssistant: Text-to-Speech (gTTS) and Speech-to-Text (Whisper).
Uses a sequential queue so TTS messages never overlap.
"""

import os
import io
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime
import shutil
try:
    from gtts import gTTS
    import whisper
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: gTTS or openai-whisper not found. Audio features disabled.")

from .config import WHISPER_MODEL


class AudioAssistant:
    """
    Handles TTS (gTTS + ffmpeg speed-up) and STT (Whisper + arecord).
    A background worker thread drains the speech queue sequentially.
    """

    def __init__(self, log_dir: Path, executor=None):
        self.log_dir = log_dir / "audio"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.executor = executor
        self.whisper_model = None
        self.last_spoken: Optional[str] = None

        # TTS queue + worker
        self.speech_queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._speech_worker, daemon=True).start()

        if AUDIO_AVAILABLE:
            try:
                print(f"Loading Whisper model ({WHISPER_MODEL})... this may take a moment.")
                self.whisper_model = whisper.load_model(WHISPER_MODEL)
                print("Whisper model loaded.")
            except Exception as e:
                print(f"Failed to load Whisper model: {e}")

    # ── TTS ───────────────────────────────────────────────────────────────────

    def speak(self, text: str):
        """Enqueue text for sequential TTS playback."""
        if not text:
            return
        self.last_spoken = text
        self.speech_queue.put(text)

    def repeat_last(self):
        """Re-speak the last spoken text."""
        if self.last_spoken:
            self.speak(self.last_spoken)
        else:
            self.speak("I haven't said anything yet.")

    def _speech_worker(self):
        """Background thread: drain speech queue one item at a time."""
        while True:
            text = self.speech_queue.get()
            if text is None:  # sentinel to stop
                break
            try:
                if not AUDIO_AVAILABLE:
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_path = fp.name

                # Generate speech
                tts = gTTS(text=text, lang='en')
                tts.save(temp_path)

                # Speed up 1.5× with ffmpeg if available
                if shutil.which("ffmpeg") is not None:
                    try:
                        fast_path = temp_path.replace(".mp3", "_fast.mp3")
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", temp_path,
                             "-filter:a", "atempo=1.5", "-vn", fast_path],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                        )
                        os.replace(fast_path, temp_path)
                    except Exception as e:
                        print(f"Audio speed adjustment failed: {e}")

                # Play (mpg123 → ffplay → warn)
                if shutil.which("mpg123") is not None:
                    subprocess.run(["mpg123", "-q", temp_path],
                                   check=False, stdin=subprocess.DEVNULL)
                elif shutil.which("ffplay") is not None:
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-hide_banner", temp_path],
                        check=False, stdin=subprocess.DEVNULL
                    )
                else:
                    print("Error: No audio player found (install mpg123 or ffmpeg).")

                os.remove(temp_path)

            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                self.speech_queue.task_done()

    # ── STT ───────────────────────────────────────────────────────────────────

    def listen(self, duration: int = 5) -> Optional[str]:
        """Record for `duration` seconds then transcribe with Whisper."""
        if not AUDIO_AVAILABLE or not self.whisper_model:
            print("Audio tools not available.")
            return None

        print(f"Listening for {duration} seconds... (Speak now)")
        timestamp = datetime.now().strftime("%H%M%S")
        wav_path = self.log_dir / f"rec_{timestamp}.wav"

        try:
            cmd = ["arecord", "-f", "cd", "-d", str(duration), str(wav_path)]
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Recording failed (arecord): {e}")
            return None

        if not wav_path.exists() or wav_path.stat().st_size < 100:
            print("Recording failed or empty.")
            return None

        print("Transcribing...")
        try:
            result = self.whisper_model.transcribe(str(wav_path))
            text = result["text"].strip()
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None

    def listen_manual_stop(self) -> Optional[str]:
        """
        Start recording; wait for the user to press Enter to stop.
        Returns transcribed text or None.
        """
        if not AUDIO_AVAILABLE or not self.whisper_model:
            print("Audio tools not available.")
            return None

        timestamp = datetime.now().strftime("%H%M%S")
        wav_path = self.log_dir / f"rec_{timestamp}.wav"

        print("\n[Recording started] Speak now... (Press Enter to stop)")
        try:
            cmd = ["arecord", "-f", "cd", str(wav_path)]
            process = subprocess.Popen(cmd,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                input()
            except EOFError:
                pass
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
            print("[Recording stopped]")
        except Exception as e:
            print(f"Recording failed: {e}")
            return None

        if not wav_path.exists() or wav_path.stat().st_size < 100:
            print("Recording failed or empty.")
            return None

        print("Transcribing...")
        try:
            result = self.whisper_model.transcribe(str(wav_path))
            text = result["text"].strip()
            print(f"You said: {text}")
            return text
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None
