
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIRECTORY = PROJECT_ROOT / "data" / "audio"
TEXT_DIRECTORY = PROJECT_ROOT / "data" / "text"


audio_files = sorted(
    file
    for file in AUDIO_DIRECTORY.iterdir()
    if file.is_file()
    and file.suffix.lower() == ".wav"
)


matched = []
missing = []


for audio_file in audio_files:

    transcript_file = (
        TEXT_DIRECTORY / f"{audio_file.stem}.txt"
    )

    if transcript_file.exists():
        matched.append(audio_file.name)
    else:
        missing.append(audio_file.name)


print(f"Audio files found: {len(audio_files)}")
print(f"Matching pairs: {len(matched)}")
print(f"Missing transcripts: {len(missing)}")


if missing:
    print("\nMissing transcripts:")
    for filename in missing:
        print(filename)


print("\nDataset check completed.")
