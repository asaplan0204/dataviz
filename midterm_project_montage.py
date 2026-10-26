
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

            if len(images) >= 40:
                break

        if len(images) >= 40:
            break

    except (json.JSONDecodeError, TypeError):
        continue

fig, axes = plt.subplots(4, 10, figsize=(16, 8))
loaded_count = 0

for ax, (event_type, url) in zip(axes.ravel(), images):
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGB")
        ax.imshow(img)
        ax.set_title(event_type, fontsize=8)
        loaded_count += 1

    except Exception as e:
        ax.text(
            0.5, 0.5,
            "Image unavailable",
            ha="center",
            va="center",
            fontsize=7
        )

    ax.axis("off")

for ax in axes.ravel()[len(images):]:
    ax.axis("off")

plt.suptitle("Bluesky Images: First 40 Images", fontsize=14)
plt.tight_layout()

plt.savefig("bluesky_montage.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Image URLs collected: {len(images)}")
print(f"Images successfully loaded: {loaded_count}")
print("Saved montage to bluesky_montage.png")
