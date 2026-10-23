"""Speech recognition extractor (audio/video -> timed transcript) with Whisper.

``transcript`` / ``transcript:<size>`` uses openai-whisper (tiny, base, small,
medium, large-v3). Audio is decoded with ffmpeg, so any container works,
including the audio track of a video.
"""
from __future__ import annotations

from omnisearch.encoders import media
from omnisearch.encoders.base import ModelUnavailable, register_extractor


class WhisperTranscriber:
    def __init__(self, size: str | None = None):
        try:
            import whisper
        except ImportError as e:
            raise ModelUnavailable("pip install openai-whisper for speech transcripts") from e
        self.name = f"transcript:whisper-{size or 'base'}"
        try:
            self.model = whisper.load_model(size or "base")
        except Exception as e:
            raise ModelUnavailable(f"could not load whisper {size or 'base'}: {e}") from e

    def transcribe(self, path) -> list[dict]:
        if not media.has_audio(path):
            return []
        audio = media.decode_audio(path, sr=16000)
        if len(audio) == 0:
            return []
        result = self.model.transcribe(audio, fp16=False)
        return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
                for s in result.get("segments", [])]

    def __call__(self, paths) -> list[list[dict]]:
        return [self.transcribe(p) for p in paths]


register_extractor("transcript", lambda arg: WhisperTranscriber(arg))
