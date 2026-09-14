from pathlib import Path
import sys


# Allow imports from Voice_Cloning
CURRENT_DIR = Path(__file__).resolve().parent
VOICE_CLONING_DIR = CURRENT_DIR.parent

if str(VOICE_CLONING_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_CLONING_DIR))


from fine_tuning.dataset import VoiceDataset


class FineTuningTrainer:
    """
    Prepares the existing audio + transcript dataset
    for XTTS-v2 fine-tuning.
    """

    def __init__(self):
        self.project_root = (
            VOICE_CLONING_DIR.parent
        )

        self.dataset = VoiceDataset(
            self.project_root
        )

    def validate_dataset(self):
        """
        Check that audio files have matching transcripts.
        """

        pairs = self.dataset.get_training_pairs()

        if not pairs:
            raise ValueError(
                "No matching audio-transcript pairs found."
            )

        missing_transcripts = []

        for audio_file in self.dataset.get_audio_files():

            transcript = self.dataset.get_transcript(
                audio_file
            )

            if not transcript:
                missing_transcripts.append(
                    audio_file.name
                )

        return pairs, missing_transcripts

    def prepare(self):
        """
        Validate and display the existing dataset.
        """

        pairs, missing_transcripts = (
            self.validate_dataset()
        )

        print(
            f"Audio files found: "
            f"{len(self.dataset.get_audio_files())}"
        )

        print(
            f"Matching audio-transcript pairs: "
            f"{len(pairs)}"
        )

        print(
            f"Missing transcripts: "
            f"{len(missing_transcripts)}"
        )

        print(
            f"Audio directory: "
            f"{self.dataset.audio_directory}"
        )

        print(
            f"Text directory: "
            f"{self.dataset.text_directory}"
        )

        print("\nSample training pairs:")

        for pair in pairs[:5]:

            print(
                f"\nAudio: {pair['audio'].name}"
            )

            print(
                f"Text: {pair['text']}"
            )

        return pairs


def main():

    trainer = FineTuningTrainer()

    trainer.prepare()


if __name__ == "__main__":
    main()
