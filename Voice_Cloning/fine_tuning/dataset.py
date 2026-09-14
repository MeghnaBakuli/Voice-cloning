from pathlib import Path


class VoiceDataset:
    """
    Connects existing audio recordings in data/audio
    with their matching transcripts in data/text.
    """

    SUPPORTED_EXTENSIONS = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a"
    }

    def __init__(self, project_root):
        self.project_root = Path(project_root)

        self.audio_directory = (
            self.project_root / "data" / "audio"
        )

        self.text_directory = (
            self.project_root / "data" / "text"
        )

    def get_audio_files(self):
        """Return all existing audio recordings."""

        if not self.audio_directory.exists():
            return []

        return sorted(
            file
            for file in self.audio_directory.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in self.SUPPORTED_EXTENSIONS
        )

    def get_transcript(self, audio_file):
        """
        Find the transcript matching an audio file.
        Example:
            recording_123.wav
            recording_123.txt
        """

        transcript_file = (
            self.text_directory
            / f"{audio_file.stem}.txt"
        )

        if not transcript_file.exists():
            return None

        try:
            return transcript_file.read_text(
                encoding="utf-8"
            ).strip()

        except OSError:
            return None

    def get_training_pairs(self):
        """
        Return audio-transcript pairs where both files exist.
        """

        pairs = []

        for audio_file in self.get_audio_files():

            transcript = self.get_transcript(
                audio_file
            )

            if transcript:
                pairs.append(
                    {
                        "audio": audio_file,
                        "text": transcript
                    }
                )

        return pairs

    def __len__(self):
        return len(self.get_training_pairs())
