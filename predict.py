from cog import BasePredictor, Input, Path
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
import os

os.environ["HF_HOME"] = "/src/.cache"
os.environ["TORCH_HOME"] = "/src/.cache"

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda"
        self.compute_type = "float16"
        hf_token = os.environ.get("HF_TOKEN")
        
        # CHANGED: Use the exact model name that we're downloading
        model_name = "Systran/faster-distil-whisper-large-v2"
        
        print(f"Loading WhisperX model: {model_name}...")
        self.whisper_model = whisperx.load_model(
            model_name,  # This now matches what's in cog.yaml
            self.device,
            compute_type=self.compute_type,
            download_root="/src/.cache"
        )
        
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device, model_cache_dir="/src/.cache"
        )
        
        if hf_token:
            self.diarization_pipeline = DiarizationPipeline(
                use_auth_token=hf_token, device=self.device
            )
        else:
            self.diarization_pipeline = None
    
    def predict(
        self,
        audio_file: Path = Input(description="Audio file to transcribe"),
        diarize: bool = Input(description="Enable speaker diarization (slower)", default=True)
    ) -> str:
        audio = whisperx.load_audio(str(audio_file))
        result = self.whisper_model.transcribe(audio, batch_size=16)
        
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        
        if diarize and self.diarization_pipeline:
            diarize_segments = self.diarization_pipeline(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        speaker_transcript = ""
        for seg in result.get("segments", []):
            speaker = seg.get("speaker", "SPEAKER_00")
            text = seg.get("text", "").strip()
            if text:
                speaker_transcript += f"\n{speaker}: {text}"
        
        return speaker_transcript.strip()
