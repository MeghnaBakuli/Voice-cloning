from pathlib import Path
import sys
import os
import shutil


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
VOICE_CLONING_DIR = CURRENT_DIR.parent
PROJECT_ROOT = VOICE_CLONING_DIR.parent

if str(VOICE_CLONING_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_CLONING_DIR))


from fine_tuning.dataset import VoiceDataset


# ============================================================
# XTTS TRAINING
# ============================================================

class FineTuningTrainer:
    """
    Fine-tunes XTTS-v2 using the existing audio + transcript
    pairs stored in data/audio and data/text.
    """

    def __init__(self):
        self.project_root = PROJECT_ROOT

        self.dataset = VoiceDataset(
            self.project_root
        )

        self.output_directory = (
            self.project_root
            / "data"
            / "voice_cloning"
            / "fine_tuned_models"
        )

        self.prepared_dataset = (
            self.output_directory
            / "dataset"
        )

    # --------------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------------

    def validate_dataset(self):
        pairs = self.dataset.get_training_pairs()

        if not pairs:
            raise ValueError(
                "No audio-transcript pairs were found."
            )

        print(
            f"Found {len(pairs)} valid audio-transcript pairs."
        )

        return pairs

    # --------------------------------------------------------
    # PREPARE XTTS DATASET
    # --------------------------------------------------------

    def prepare_dataset(self, pairs):
        """
        Convert the existing dataset into the format expected
        by the official XTTS Coqui trainer.
        """

        wav_directory = (
            self.prepared_dataset / "wavs"
        )

        wav_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata = []

        print("\nPreparing XTTS dataset...")

        for index, pair in enumerate(pairs):

            source_audio = pair["audio"]
            transcript = pair["text"]

            output_name = (
                f"recording_{index:04d}"
                f"{source_audio.suffix.lower()}"
            )

            destination_audio = (
                wav_directory / output_name
            )

            shutil.copy2(
                source_audio,
                destination_audio
            )

            metadata.append(
                f"wavs/{output_name}|"
                f"{transcript}|"
                f"speaker_001"
            )

        # 85% training / 15% evaluation
        split_index = max(
            1,
            int(len(metadata) * 0.85)
        )

        train_metadata = metadata[:split_index]
        eval_metadata = metadata[split_index:]

        if not eval_metadata:
            eval_metadata = train_metadata[-1:]
            train_metadata = train_metadata[:-1]

        train_csv = (
            self.prepared_dataset
            / "metadata_train.csv"
        )

        eval_csv = (
            self.prepared_dataset
            / "metadata_eval.csv"
        )

        train_csv.write_text(
            "\n".join(train_metadata),
            encoding="utf-8"
        )

        eval_csv.write_text(
            "\n".join(eval_metadata),
            encoding="utf-8"
        )

        print(
            f"Training samples: {len(train_metadata)}"
        )

        print(
            f"Evaluation samples: {len(eval_metadata)}"
        )

        print(
            f"Dataset prepared at: "
            f"{self.prepared_dataset}"
        )

        return train_csv, eval_csv

    # --------------------------------------------------------
    # START XTTS FINE-TUNING
    # --------------------------------------------------------

    def train(
        self,
        epochs=2,
        batch_size=1,
        grad_accumulation=4
    ):

        pairs = self.validate_dataset()

        train_csv, eval_csv = (
            self.prepare_dataset(pairs)
        )

        print("\nStarting XTTS-v2 fine-tuning...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(
            f"Gradient accumulation: "
            f"{grad_accumulation}"
        )

        from TTS.demos.xtts_ft_demo.utils.gpt_train import (
            train_gpt
        )

        (
            config_file,
            original_checkpoint,
            tokenizer_file,
            trainer_output,
            speaker_reference
        ) = train_gpt(
            language="en",
            num_epochs=epochs,
            batch_size=batch_size,
            grad_acumm=grad_accumulation,
            train_csv=str(train_csv),
            eval_csv=str(eval_csv),
            output_path=str(
                self.output_directory
            ),
            max_audio_length=255995
        )

        print("\n========================================")
        print("XTTS FINE-TUNING COMPLETED")
        print("========================================")

        print(
            f"\nTrainer output:\n{trainer_output}"
        )

        print(
            f"\nSpeaker reference:\n{speaker_reference}"
        )

        return trainer_output


# ============================================================
# MAIN
# ============================================================

def main():

    trainer = FineTuningTrainer()

    # Small initial test run.
    trainer.train(
        epochs=2,
        batch_size=1,
        grad_accumulation=4
    )


if __name__ == "__main__":
    main()

