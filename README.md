# WhisperX Replicate Model

This model uses WhisperX for transcription, speaker diarization, and optional translation via GPT-4o. 

## Inputs
- `audio`: Audio file input
- `original_language`: (Optional) Source language. Autodetect if left blank.
- `translate_to`: Target language (default is English)
- `diarize`: Whether to apply speaker diarization (default is True)

## Outputs
- Transcribed text (with or without diarization)
- Translated segments if requested
