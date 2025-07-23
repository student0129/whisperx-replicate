# --------------------
# File: predict.py
# --------------------
from cog import BasePredictor, Input, Path
import whisper
import os

class Predictor(BasePredictor):
    def setup(self):
        self.model = whisper.load_model("large")

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        language: str = Input(description="Language code (or leave blank to auto-detect)", default=None),
        translate: bool = Input(description="Translate to English", default=False),
    ) -> str:
        result = self.model.transcribe(
            str(audio), language=language, task="translate" if translate else "transcribe"
        )
        return result["text"]
