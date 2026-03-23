import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import json
from sklearn.cluster import KMeans
from rembg import remove

# 1. Load Model and Classes
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model('flower_model.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

# 2. Helper to find closest color name
COLORS = {
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

def get_color_name(rgb_tuple):
    min_dist = float('inf')
    closest_name = "Unknown"
    for name, color in COLORS.items():
        # Calculate Euclidean distance
        dist = sum((a - b) ** 2 for a, b in zip(rgb_tuple, color))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

# 3. Main Streamlit App Layout & Config
st.set_page_config(page_title="Flower Lens AI", page_icon="🌸", layout="wide")

# Massive Aesthetic Overhaul with Custom CSS injected directly
st.markdown("""
<style>
/* Mimic the background elements and blobs */
.stApp {
    background-color: #f0f4f8;
    background-image: 
        radial-gradient(circle at 10% -10%, rgba(255, 154, 158, 0.4) 0%, rgba(254, 207, 239, 0) 50vw),
        radial-gradient(circle at 110% 120%, rgba(161, 196, 253, 0.4) 0%, rgba(194, 233, 251, 0) 60vw);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
    color: #333;
}

/* Glassmorphism styling around main block */
.block-container {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    border-radius: 20px;
    padding: 40px !important;
    max-width: 1100px;
    margin-top: 40px;
}

/* Title & Subtitle Matching */
h1 {
    text-align: center;
    color: #ff4b4b !important;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
    margin-bottom: 5px !important;
    padding-bottom: 0px !important;
}

.custom-subtitle {
    text-align: center;
    color: #666;
    font-size: 1.15rem;
    margin-bottom: 40px;
    margin-top: 0px !important;
}

/* Streamlit Native Tabs Styled to Match */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.05rem;
    font-weight: 600;
    color: #64748b;
    border-radius: 4px 4px 0 0;
    transition: color 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #ff4b4b !important;
    border-bottom-color: #ff4b4b !important;
}

/* Metrics Cards Styled specifically */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.9);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    border: 1px solid rgba(255,255,255,0.8);
}
[data-testid="stMetricLabel"] {
    font-size: 0.95rem;
    text-transform: uppercase;
    font-weight: 600;
    color: #64748b;
}
[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 800;
    color: #0f172a;
}

/* Specific elements injected via markdown */
.glass-card {
    background: rgba(255, 255, 255, 0.9);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}
.glass-card h4 {
    margin-top: 0;
    color: #475569;
    font-size: 1.1rem;
}
.swatch-container {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-top: 15px;
}
.swatch {
    width: 70px;
    height: 70px;
    border-radius: 14px;
    border: 3px solid #fff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.color-info h2 {
    margin: 0 0 5px 0;
    font-size: 1.5rem;
    color: #1e293b;
    font-weight: 800;
}
.color-info p {
    margin: 0;
    color: #64748b;
    font-family: monospace;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🌸 Flower Lens AI")
st.markdown('<p class="custom-subtitle">Upload a beautiful flower image for instant AI-powered classification, color extraction, and visual analysis.</p>', unsafe_allow_html=True)

try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error("Model not found. Please run `train.py` first to generate the improved model.")
    st.stop()

# Layout Configuration: Left Uploader vs Right Results
col_upload, col_analysis = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown("### 📸 Image Upload")
    uploaded_file = st.file_uploader("Select Flower Image (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        st.image(image, use_container_width=True, caption="Preview")
    else:
        st.info("Or drag and drop your image here.")

if uploaded_file is not None:
    with col_analysis:
        tab1, tab2, tab3 = st.tabs(["✨ AI Classification", "🎨 Color Intelligence", "📈 Spectra Analysis"])

        with tab1:
            st.markdown("### Network Prediction")
            with st.spinner("AI is analyzing the flower..."):
                # Preprocess for CNN
                resized_img = tf.image.resize(img_array, [224, 224])
                input_arr = tf.expand_dims(resized_img, 0) # Create a batch
    
                # Predict
                predictions = model.predict(input_arr)
                score = tf.nn.softmax(predictions[0])
                predicted_class = class_names[np.argmax(score)].title()
                confidence = float(100 * np.max(score))

            met1, met2 = st.columns(2)
            with met1:
                st.metric(label="Predicted Species", value=predicted_class)
            with met2:
                st.metric(label="Confidence Rating", value=f"{confidence:.2f}%")

            if confidence > 80:
                st.success(f"High confidence match! The model strongly believes this is a **{predicted_class}**.")
            else:
                st.warning(f"Moderate confidence. This looks like a **{predicted_class}**, but it could be another variety.")

        with tab2:
            st.markdown("### Chromatic Profile")
            with st.spinner("Isolating flower with advanced background removal..."):
                img_no_bg = remove(image)
                img_no_bg_array = np.array(img_no_bg)
                mask = img_no_bg_array[:, :, 3] > 0
                flower_pixels = img_no_bg_array[mask][:, :3]

            if len(flower_pixels) > 0:
                kmeans = KMeans(n_clusters=1, n_init=10)
                kmeans.fit(flower_pixels)
                dominant_color = kmeans.cluster_centers_[0].astype(int)
                color_name = get_color_name(dominant_color)
                
                iso_col, spec_col = st.columns([1, 1])
                with iso_col:
                    st.image(img_no_bg, use_container_width=True, caption="Background Removed")
                
                with spec_col:
                    # Glass card styling for swatch
                    swatch_html = f'''
                    <div class="glass-card">
                        <h4>Dominant Hue</h4>
                        <div class="swatch-container">
                            <div class="swatch" style="background-color:rgb({dominant_color[0]},{dominant_color[1]},{dominant_color[2]});"></div>
                            <div class="color-info">
                                <h2>{color_name}</h2>
                                <p>RGB: {dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]}</p>
                            </div>
                        </div>
                    </div>
                    '''
                    st.markdown(swatch_html, unsafe_allow_html=True)
            else:
                st.warning("Could not isolate the flower from the background.")

        with tab3:
            st.markdown("### Original Image RGB Histogram")
            # Interactive Streamlit line chart
            hist_r = cv2.calcHist([img_array], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_array], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_array], [2], None, [256], [0, 256]).flatten()
            
            chart_data = pd.DataFrame({
                'Red (R)': hist_r,
                'Green (G)': hist_g,
                'Blue (B)': hist_b
            })
            
            st.line_chart(chart_data, color=["#FF0000", "#00FF00", "#0000FF"])
else:
    with col_analysis:
        st.markdown("<div style='margin-top:100px;text-align:center;color:#94a3b8;font-style:italic;font-size:1.1rem;'>Upload an image to see prediction and analysis results!</div>", unsafe_allow_html=True)
