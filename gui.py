import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import scipy.spatial as spatial

# --- APP CONFIG ---
st.set_page_config(page_title="Flower AI Analyzer", layout="wide")
st.title("🌸 Flower Classifier & Color Identifier")


# --- COLOR NAMING LOGIC ---
def match_color_name(rgb_triplet):
    # Dictionary of common flower colors
    color_db = {
        "Red": (200, 20, 20),
        "Dark Red": (139, 0, 0),
        "Light Pink": (255, 182, 193),
        "Hot Pink": (255, 105, 180),
        "Deep Pink": (255, 20, 147),
        "Coral": (255, 127, 80),
        "Magenta / Fuchsia": (255, 0, 255),
        "Violet": (238, 130, 238),
        "Indigo": (75, 0, 130),
        "Orchid / Medium Purple": (168, 66, 164),
        "Burnt Orange": (204, 85, 0),
        "Shadowed Yellow": (158, 128, 22),
        "Yellow": (245, 190, 10),
        "Orange": (240, 100, 10),
        "Purple": (140, 50, 140),
        "Blue": (30, 60, 200),
        "Light Blue": (120, 180, 230),
        "Green": (50, 160, 50),
        "Dark Green": (20, 80, 20),
        "White": (214, 215, 207),
        "Black": (20, 20, 20),
        "Brown": (120, 70, 30),
        "Gray": (130, 130, 130)
    }

    names = list(color_db.keys())
    values = list(color_db.values())

    # Find the closest color using Euclidean distance
    tree = spatial.KDTree(values)
    dist, index = tree.query(rgb_triplet)
    return names[index]


# Load resources
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('flower_model.h5')
    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes


try:
    model, CLASS_NAMES = load_resources()
except:
    st.error("Please run train.py first to generate the model!")
    st.stop()


def get_dominant_color(image, k=3):
    # Focus on the center of the image
    h, w, _ = image.shape
    cp = image[h // 4:3 * h // 4, w // 4:3 * w // 4]
    img = cv2.resize(cp, (50, 50))
    pixels = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    # Get the color of the largest cluster
    counts = np.bincount(kmeans.labels_)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

    # CONVERT TO REGULAR PYTHON INTEGERS (Fixes the np.int64 issue)
    return [int(c) for c in dominant_color]


def plot_histogram(image):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col, linewidth=2)
    ax.set_title('RGB Color Intensity Distribution')
    ax.set_xlabel('Pixel Intensity (0-255)')
    ax.set_ylabel('Number of Pixels')
    ax.set_facecolor('#f9f9f9')
    return fig


# --- UI ---
uploaded_file = st.file_uploader("Upload a flower photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img_rgb, caption='Target Image', use_container_width=True)

        # Classification
        # --- IMPROVED CLASSIFICATION LOGIC ---
        input_img = cv2.resize(img_rgb, (224, 224)) / 255.0
        prediction = model.predict(np.expand_dims(input_img, axis=0))

        # Get the highest probability score and its index
        conf_score = np.max(prediction)
        label_index = np.argmax(prediction)

        # Set a strict threshold (0.80 = 80% certainty)
        THRESHOLD = 0.80

        if conf_score < THRESHOLD:
            st.error("❌ **Error: Object Not Identified**")
            st.info("The system is not confident that this is a flower. Please upload a clearer image of a flower.")
        else:
            label = CLASS_NAMES[label_index]
            st.success(f"### Flower Type: {label}")
            st.write(f"**Confidence Score:** {conf_score * 100:.2f}%")
            st.progress(float(conf_score))

            # ... (rest of your color display code)
    with col2:
        # Dominant Color & Name
        st.subheader("🎨 Color Detection")

        # Get dominant color in RGB
        dom_bgr = get_dominant_color(img_bgr)
        dom_rgb = dom_bgr[::-1]  # Flip BGR list to RGB

        color_name = match_color_name(dom_rgb)

        # UI for color display
        c1, c2 = st.columns([1, 2])
        with c1:
            # Create a small color square swatch
            color_swatch = np.zeros((100, 100, 3), dtype=np.uint8)
            color_swatch[:] = dom_rgb
            st.image(color_swatch)
        with c2:
            st.markdown(f"#### Detected: **{color_name}**")
            # Clean tuple display
            st.code(f"RGB Value: {tuple(dom_rgb)}")

        # Histogram
        st.subheader("📊 Color Distribution Graph")
        st.pyplot(plot_histogram(img_bgr))