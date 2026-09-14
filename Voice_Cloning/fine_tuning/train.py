
from pathlib import Path
import sys


# Allow imports from Voice_Cloning
CURRENT_DIR = Path(__file__).resolve().parent
VOICE_CLONING_DIR = CURRENT_DIR.parent

if str(VOICE_CLONING_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_CLONING_DIR))


from config import DATASET_DIRECTORY, TRAINED_MODELS_DIRECTORY


class FineTuningTrainer:
    """
    XTTS-v2 fine-tuning trainer.

    This module prepares the training configuration
    and launches the Coqui XTTS training process.
    """

    def __init__(self, user_id):
        self.user_id = user_id

        self.dataset_directory = (
            Path(DATASET_DIRECTORY) / user_id
        )

        self.output_directory = (
            Path(TRAINED_MODELS_DIRECTORY) / user_id
        )

    def validate_dataset(self):
        """
        Check whether the user's dataset exists
        and contains audio files.
        """

        if not self.dataset_directory.exists():
            raise FileNotFoundError(
                f"Dataset not found for user: {self.user_id}"
            )

        audio_extensions = {
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a"
        }

        audio_files = [
            file
            for file in self.dataset_directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            raise ValueError(
                f"No audio files found for user: {self.user_id}"
            )

        return sorted(audio_files)

    def prepare_output_directory(self):
        """
        Create the user's model output directory.
        """

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        return self.output_directory

    def prepare(self):
        """
        Validate the dataset and prepare the
        fine-tuning output directory.
        """

        audio_files = self.validate_dataset()
        output_directory = (
            self.prepare_output_directory()
        )

        print(
            f"User: {self.user_id}"
        )

        print(
            f"Audio files found: {len(audio_files)}"
        )

        print(
            f"Dataset: {self.dataset_directory}"
        )

        print(
            f"Output: {output_directory}"
        )

        return {
            "user_id": self.user_id,
            "audio_files": audio_files,
            "dataset_directory": self.dataset_directory,
            "output_directory": output_directory
        }


def main():

    if len(sys.argv) < 2:
        print(
            "Usage: python train.py <user_id>"
        )
        return

    user_id = sys.argv[1]

    trainer = FineTuningTrainer(
        user_id
    )

    trainer.prepare()


if __name__ == "__main__":
    main()
