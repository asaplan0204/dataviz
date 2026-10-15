
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = Path(
    "/Users/aspen_saplan/Documents/dataviz/rice_dataset"
)

df = pd.read_csv(DATASET_DIR / "rice_dataset.csv")

OUTPUT_DIR = DATASET_DIR / "aligned_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 128


def align_rice(image_path, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try both polarities: light grain on dark background
    # or dark grain on light background.
    _, mask_light = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, mask_dark = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    candidates = []

    for mask in (mask_light, mask_dark):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 10:
                candidates.append((area, largest))

    if not candidates:
        return False

    # The largest detected foreground object is assumed to be the grain.
    contour = max(candidates, key=lambda item: item[0])[1]


    # Estimate the grain's long axis using PCA
    points = contour.reshape(-1, 2).astype(np.float32)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        points, mean=None
    )

    # Direction of the longest axis
    vx, vy = eigenvectors[0]

    # Angle relative to the horizontal
    angle = np.degrees(np.arctan2(vy, vx))

    # Rotate the long axis to vertical.
    # Normalize the correction to the range [-90, 90].
    rotation = angle - 90
    rotation = (rotation + 90) % 180 - 90

    cx, cy = mean[0]

    matrix = cv2.getRotationMatrix2D(
        (float(cx), float(cy)),
        rotation,
        1.0
    )

    matrix = cv2.getRotationMatrix2D(
        (cx, cy), rotation, 1.0
    )

    rotated = cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # Rotate the binary mask too, then crop around the grain.
    rotated_mask = cv2.warpAffine(
        cv2.drawContours(
            np.zeros_like(gray), [contour], -1, 255, -1
        ),
        matrix, (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    points = cv2.findNonZero(rotated_mask)

    if points is None:
        return False

    x, y, bw, bh = cv2.boundingRect(points)
    grain = rotated[y:y + bh, x:x + bw]

    # Fit the grain inside a square without distorting its shape.
    gh, gw = grain.shape[:2]
    scale = min((SIZE - 8) / gw, (SIZE - 8) / gh)

    new_w = max(1, round(gw * scale))
    new_h = max(1, round(gh * scale))

    grain = cv2.resize(
        grain, (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)

    x0 = (SIZE - new_w) // 2
    y0 = (SIZE - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = grain

    return cv2.imwrite(str(output_path), canvas)


new_paths = []

for _, row in df.iterrows():
    source = DATASET_DIR / row["image_path"]
    destination = OUTPUT_DIR / row["filename"]

    if align_rice(source, destination):
        new_paths.append(str(destination))
    else:
        new_paths.append(None)
        print(f"Could not align: {source}")

df["aligned_path"] = new_paths

df.to_csv(
    DATASET_DIR / "rice_dataset_aligned.csv",
    index=False
)

print("Alignment finished.")
print("Successful:", df["aligned_path"].notna().sum())
print("Output:", OUTPUT_DIR)
