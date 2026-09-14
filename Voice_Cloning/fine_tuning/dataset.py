
from pathlib import Path
import json


class VoiceDataset:
    """
    Handles audio files and metadata for voice fine-tuning.

    Expected dataset structure:

    data/
    └── voice_cloning/
        └── datasets/
            └── <user_id>/
                ├── audio1.wav
                ├── audio2.wav
                └── metadata.json
    """

    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self, dataset_directory, user_id):
        self.dataset_directory = Path(dataset_directory)
        self.user_id = user_id

        self.user_directory = self.dataset_directory / user_id
        self.metadata_file = self.user_directory / "metadata.json"

    def get_audio_files(self):
        """Return all supported audio files for the user."""

        if not self.user_directory.exists():
            return []

        audio_files = [
            file
            for file in self.user_directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        return sorted(audio_files)

    def get_metadata(self):
        """Load metadata.json if it exists."""

        if not self.metadata_file.exists():
            return []

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as file:
                return json.load(file)

        except (json.JSONDecodeError, OSError):
            return []

    def create_metadata(self, transcript_map=None):
        """
        Create basic metadata for the available audio files.

        transcript_map:
            Optional dictionary mapping filename -> transcript.
        """

        transcript_map = transcript_map or {}

        metadata = []

        for audio_file in self.get_audio_files():
            metadata.append(
                {
                    "audio_file": audio_file.name,
                    "text": transcript_map.get(audio_file.name, ""),
                }
            )

        self.user_directory.mkdir(parents=True, exist_ok=True)

        with open(self.metadata_file, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        return metadata

    def __len__(self):
        return len(self.get_audio_files())
