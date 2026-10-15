
from pathlib import Path
import shutil
import pandas as pd

SOURCE_DIR = Path("/Users/aspen_saplan/Documents/dataviz/Rice_Image_Dataset")

OUTPUT_DIR = Path("/Users/aspen_saplan/Documents/dataviz/Rice_Dataset")
IMAGE_DIR = OUTPUT_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

FOLDER_LABELS = {
    "Basmati": "Basmati",
    "Jasmine": "Jasmine",
    "Arborio": "Arborio",
    "Ipsala": "Ipsala",
    "Karacadag": "Karacadag"
}

MAX_IMAGES = 50
records = []

for folder_name, label in FOLDER_LABELS.items():
    folder_path = SOURCE_DIR / folder_name

    if not folder_path.is_dir():
        print(f"Folder not found: {folder_path}")
        continue

    image_files = sorted(
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg"]
    )

    selected_images = image_files[:MAX_IMAGES]

    for i, image_path in enumerate(selected_images, start=1):
        # Unique output filename, even if original names repeat
        new_filename = f"{label}_{i:03d}{image_path.suffix.lower()}"
        destination = IMAGE_DIR / new_filename

        shutil.copy2(image_path, destination)

        records.append({
            "filename": new_filename,
            "label": label,
            "original_path": str(image_path),
            "image_path": f"images/{new_filename}"
        })

    print(f"{label}: selected {len(selected_images)} images")

df = pd.DataFrame(records)
csv_path = OUTPUT_DIR / "rice_dataset.csv"
df.to_csv(csv_path, index=False)

print(f"\nTotal images: {len(df)}")
print(f"Dataset CSV: {csv_path}")
print(df["label"].value_counts())
