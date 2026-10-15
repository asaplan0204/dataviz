import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_image_comparison import image_comparison

st.set_page_config(layout="wide")
st.title("Rice Data")

BASE_DIR = Path(__file__).resolve().parent
folder = BASE_DIR / "Rice_Dataset" / "aligned_images"

tab1, tab2, tab3 = st.tabs(["Rice Image Montage", "Rice Image Intensit", "Rice Image Occlusion Visualization"])

with tab1:
    st.subheader("Rice Image Montage")

    all_images = sorted(folder.glob("*.jpg"))

    # Get rice labels from filenames
    rice_types = sorted(set(img.name.split("_")[0] for img in all_images))

    # Select 5 rice varieties
    selected_types = st.multiselect(
        "Select 5 rice varieties",
        options=rice_types,
        default=rice_types[:5]
    )

    # Select the first 8 images of each variety
    images = []
    for rice_type in selected_types[:5]:
        type_images = [img for img in all_images if img.name.startswith(rice_type + "_")]
        images.extend(type_images[:8])

    # Display montage
    fig, axes = plt.subplots(4, 10, figsize=(12, 5))

    for ax, img_path in zip(axes.ravel(), images):
        ax.imshow(Image.open(img_path))
        ax.set_title(img_path.name.split("_")[0], fontsize=8)
        ax.axis("off")

    for ax in axes.ravel()[len(images):]:
        ax.axis("off")

    plt.suptitle("First 8 Images of Each Rice Variety")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Rice Image Intensity")

    images = sorted(folder.glob("*.jpg"))

    img_path = st.selectbox(
        "Select an image",
        images,
        format_func=lambda x: x.name
    )

    img = np.array(Image.open(img_path).convert("RGB"))

    df = pd.DataFrame({
        "Red": img[..., 0].ravel(),
        "Green": img[..., 1].ravel(),
        "Blue": img[..., 2].ravel()
    })

    df = df.melt(
        var_name="Channel",
        value_name="Pixel Intensity"
    )

    fig1 = px.histogram(
        df,
        x="Pixel Intensity",
        color="Channel",
        color_discrete_map={
            "Red": "red",
            "Green": "green",
            "Blue": "blue"
        },
        nbins=50,
        barmode="overlay",
        opacity=0.5,
        title="RGB Pixel Intensity Distribution",
        labels={"count": "Frequency"}
    )

    fig1.update_layout(
        xaxis_title="Pixel Intensity (0–255)",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig1, use_container_width=True)

with tab3:
    st.subheader("Rice Image Occlusion Visualization")
    st.set_page_config(
        page_title="Rice Image Visualization",
        layout="wide"
    )

    # Load images and labels
    image_paths = sorted(folder.glob("*.jpg"))
    size = (32, 32)

    X, y = [], []

    for path in image_paths:
        label = path.stem.rsplit("_", 1)[0]
        img = Image.open(path).convert("L").resize(size)
        X.append(np.array(img).ravel())
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Train a simple rice classifier
    @st.cache_resource
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, accuracy_score(y_test, model.predict(X_test))

    model, accuracy = train_model(X, y)

    st.write(f"**Test accuracy:** {accuracy:.1%}")

    # Select an image
    selected_path = st.selectbox(
        "Select a rice image",
        image_paths,
        format_func=lambda p: p.name
    )

    img = np.array(
        Image.open(selected_path).convert("L").resize(size)
    )

    # Occlusion sensitivity
    def occlude(model, img, cls, patch=3):
        base = model.predict_proba([img.ravel()])[0][cls]
        sal = np.zeros(img.shape)
        r = patch // 2

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                temp = img.copy()
                temp[
                    max(0, i-r):min(img.shape[0], i+r+1),
                    max(0, j-r):min(img.shape[1], j+r+1)
                ] = 0

                probability = model.predict_proba(
                    [temp.ravel()]
                )[0][cls]

                sal[i, j] = base - probability

        return sal

    # Predict selected image's class
    prediction = model.predict([img.ravel()])[0]
    class_index = list(model.classes_).index(prediction)

    st.write(f"**Predicted rice variety:** {prediction}")

    with st.spinner("Generating occlusion map..."):
        saliency = occlude(model, img, class_index, patch=3)

    original_image = Image.fromarray(img).convert("RGB")

    # Convert saliency map to a colored heatmap image
    saliency_norm = (
        (saliency - saliency.min()) /
        (saliency.max() - saliency.min() + 1e-8)
    )

    saliency_rgb = (plt.cm.plasma(saliency_norm)[:, :, :3] * 255).astype(np.uint8)
    saliency_image = Image.fromarray(saliency_rgb)

    st.subheader("Original vs. Occlusion Saliency")

    image_comparison(
        img1=original_image,
        img2=saliency_image,
        label1="Original Rice Image",
        label2="Occlusion Saliency Map",
        width=700
    )
