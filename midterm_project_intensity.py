import pandas as pd
import json
import re
import requests
import numpy as np
from PIL import Image
from io import BytesIO

df = pd.read_csv(
    "bluesky_all_events.csv",
    usecols=["event_type", "image_urls"]
)

df_image = df[
    (df["event_type"] == "reply")
    & df["image_urls"].notna()
    & (df["image_urls"].str.strip() != "")
    & (df["image_urls"] != "[]")
]

images = []

for _, row in df_image.iterrows():
    try:
        urls = json.loads(row["image_urls"])

        if not isinstance(urls, list):
            continue

        for value in urls:
            if not isinstance(value, str):
                continue

            match = re.search(r"\]\((https?://[^)]+)\)", value)
            url = match.group(1) if match else value.strip()

            if url.startswith("https://"):
                images.append((row["event_type"], url))

            if len(images) >= 40:
                break

        if len(images) >= 40:
            break

    except (json.JSONDecodeError, TypeError):
        continue

rgb_data = []
successful_images = 0

for image_number, (event_type, url) in enumerate(images, start=1):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        with Image.open(BytesIO(response.content)) as img:
            img = img.convert("RGB")
            img.thumbnail((1000, 1000))

            pixels = np.asarray(img)

            for channel_index, channel_name in enumerate(
                ["Red", "Green", "Blue"]
            ):
                counts, _ = np.histogram(
                    pixels[:, :, channel_index],
                    bins=256,
                    range=(0, 256)
                )

                for intensity, count in enumerate(counts):
                    rgb_data.append({
                        "image_number": image_number,
                        "event_type": event_type,
                        "Channel": channel_name,
                        "Intensity": intensity,
                        "Pixel_Count": int(count)
                    })

        successful_images += 1

    except Exception as e:
        print(f"Skipping image {image_number}: {e}")

rgb_df = pd.DataFrame(rgb_data)
rgb_df.to_csv("bluesky_reply_rgb_intensity.csv", index=False)

print(f"Image URLs collected: {len(images)}")
print(f"Images successfully processed: {successful_images}")
print(f"Histogram rows saved: {len(rgb_df)}")
print("Saved to bluesky_reply_rgb_intensity.csv")