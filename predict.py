# --------------------
# File: predict.py
# --------------------

from cog import BasePredictor, Input, Path
from typing import Optional, Dict, Any
import whisperx
import torch
import tempfile
import os

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        self.base_model = whisperx.load_model("large-v2", device=self.device, compute_type=self.compute_type)

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        original_language: Optional[str] = Input(
            description="Input language code (e.g., 'en', 'fr', 'fa'). Leave blank to autodetect.",
            default=None
        ),
        translate_to: str = Input(
            description="Target language for translation",
            default="en"
        ),
        diarize: bool = Input(
            description="Enable speaker diarization",
            default=True
        )
    ) -> Dict[str, Any]:
        # Load base model
        model = self.base_model
        # Transcribe
        audio_path = str(audio)
        transcription = model.transcribe(audio_path)

        # Diarization
        if diarize:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=self.device)
            diarize_segments = diarize_model(audio_path)
            transcription = whisperx.merge_text_diarization(transcription["segments"], diarize_segments)

        # Optional translation (manual, not WhisperX-native)
        if translate_to and translate_to != original_language:
            translated = self._translate_segments(transcription["segments"], target_lang=translate_to)
            return {"translated_segments": translated, "language_detected": transcription["language"]}
        
        return {"segments": transcription["segments"], "language_detected": transcription["language"]}

    def _translate_segments(self, segments, target_lang: str):
        from openai import OpenAI
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        full_text = "\n".join([seg["text"] for seg in segments])
        prompt = f"""You are a professional interpreter. Translate the following spoken conversation into {target_lang}, preserving meaning, tone, and speaker intent.
            Keep formatting simple and readable. Ignore filler words. Do not summarize—translate everything.
            Conversation:
            {full_text}
            """

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
