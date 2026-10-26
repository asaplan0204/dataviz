import pandas as pd
import json
import re
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

df = pd.read_csv("bluesky_all_events.csv")

df_image = df[df["event_type"] == "reply"]

df_image = df_image[
    df_image["image_urls"].notna()
    & (df_image["image_urls"].str.strip() != "")
    & (df_image["image_urls"] != "[]")
]

images = []

for _, row in df_image.iterrows():
    try:
        urls = row["image_urls"]

        if isinstance(urls, str):
            urls = json.loads(urls)

        if not isinstance(urls, list):
            continue

        for value in urls:
            if not isinstance(value, str):
                continue

            match = re.search(r"\]\((https?://[^)]+)\)", value)
            url = match.group(1) if match else value.strip()

            if url.startswith("https://"):
                images.append((row["event_type"], url))

            if len(images) >= 10:
                break

        if len(images) >= 10:
            break

    except (json.JSONDecodeError, TypeError):
        continue
rgb_data = []

for image_number, (event_type, url) in enumerate(images, start=1):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        pixels = list(img.getdata())

        for red, green, blue in pixels:
            rgb_data.append({
                "image_number": image_number,
                "event_type": event_type,
                "Red": red,
                "Green": green,
                "Blue": blue
            })

    except Exception as e:
        print(f"Skipping image {image_number}: {e}")

rgb_df = pd.DataFrame(rgb_data)
rgb_df.to_csv("bluesky_reply_rgb_intensity.csv", index=False)

print(f"Images collected: {len(images)}")
print(f"Images successfully processed: {rgb_df['image_number'].nunique() if not rgb_df.empty else 0}")
print(f"Total pixels processed: {len(rgb_df)}")
print("Saved to bluesky_reply_rgb_intensity.csv")