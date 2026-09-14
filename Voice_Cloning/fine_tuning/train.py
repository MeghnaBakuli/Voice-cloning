
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio


# Allow imports from Voice_Cloning
CURRENT_DIR = Path(__file__).resolve().parent
VOICE_CLONING_DIR = CURRENT_DIR.parent

if str(VOICE_CLONING_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_CLONING_DIR))

from config import DATASET_DIRECTORY, TRAINED_MODELS_DIRECTORY


class AudioDataset(Dataset):
    """Simple audio dataset for testing the fine-tuning pipeline."""

    def __init__(self, audio_files):
        self.audio_files = audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_path = self.audio_files[index]

        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert stereo audio to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sample_rate


def find_audio_files(user_id):
    """Find all supported audio files for a user."""

    user_directory = Path(DATASET_DIRECTORY) / user_id

    if not user_directory.exists():
        return []

    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    return sorted(
        file
        for file in user_directory.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    )


def test_dataset(user_id):
    """Test whether the user's audio dataset can be loaded."""

    audio_files = find_audio_files(user_id)

    print(f"Audio files found: {len(audio_files)}")

    if not audio_files:
        print("No audio files found.")
        return

    dataset = AudioDataset(audio_files)

    for index in range(min(3, len(dataset))):
        waveform, sample_rate = dataset[index]

        print(
            f"{audio_files[index].name}: "
            f"shape={tuple(waveform.shape)}, "
            f"sample_rate={sample_rate}"
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <user_id>")
        return

    user_id = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Dataset directory: {DATASET_DIRECTORY}")
    print(f"Model directory: {TRAINED_MODELS_DIRECTORY}")

    test_dataset(user_id)


if __name__ == "__main__":
    main()
