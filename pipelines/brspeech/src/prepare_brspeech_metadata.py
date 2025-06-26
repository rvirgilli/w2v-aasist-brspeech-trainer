import os
import csv
from pathlib import Path

# Dataset paths inside container
REAL_BASE = "/data/real"
SPOOF_BASE = "/data/spoof"
OUTPUT_DIR = "metadata"

tts_models = ['f5tts', 'fish-speech', 'toucantts', 'xtts', 'yourtts']
splits = ['train', 'dev', 'test']

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in splits:
    index = []
    
    # Process real audio files
    real_csv_path = os.path.join(REAL_BASE, f"{split}.csv")
    if os.path.exists(real_csv_path):
        with open(real_csv_path, newline='') as f:
            reader = csv.reader(f, delimiter='|')
            next(reader)  # Skip header
            for row in reader:
                audio_path = os.path.join(REAL_BASE, row[0])
                if os.path.exists(audio_path):
                    index.append((audio_path, 'bonafide', 'bonafide'))
                else:
                    print(f"⚠️  Real audio file not found: {audio_path}")
    
    # Process synthetic audio files
    for model in tts_models:
        spoof_path = os.path.join(SPOOF_BASE, model, split)
        if os.path.exists(spoof_path):
            for dirpath, _, filenames in os.walk(spoof_path):
                for filename in filenames:
                    if filename.endswith(".flac"):
                        file_path = os.path.join(dirpath, filename)
                        index.append((file_path, 'spoof', model))
    
    # Write metadata CSV
    output_csv = os.path.join(OUTPUT_DIR, f"{split}.csv" if split != 'dev' else 'val.csv')
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "label", "synth"])
        writer.writerows(index)
    
    print(f"✅ Generated {output_csv} with {len(index)} samples")

print(f"✅ Metadata generation complete!") 