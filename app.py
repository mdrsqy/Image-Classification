import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ------------------- Konfigurasi Streamlit -------------------
st.set_page_config(page_title="Vegetable Classifier", page_icon="🥦", layout="wide")

# ---------------------- Cek pyarrow --------------------------
pyarrow_available = True
try:
    import pyarrow
except ModuleNotFoundError:
    pyarrow_available = False

# ------------------------ CSS Styling ------------------------
st.markdown("""
    <style>
    .header-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 30px;
        gap: 30px;
    }
    .identity-box {
        flex: 1;
        background-color: #e8f5e9;
        border-left: 6px solid #4CAF50;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .identity-icon {
        font-size: 36px;
        color: #388e3c;
    }
    .identity-text {
        font-size: 15px;
        line-height: 1.5;
        color: #2e7d32;
    }
    .center-title {
        text-align: center;
        font-size: 32px;
        color: #4CAF50;
        font-weight: bold;
    }
    .center-subtitle {
        text-align: center;
        color: #555;
        font-size: 16px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------- Header Identitas ---------------------
st.markdown("""
    <div class="header-container">
        <div class="identity-box">
            <div class="identity-icon">👤</div>
            <div class="identity-text">
                <strong>Nama:</strong> Muhammad Dzaky Ramdani Syakur<br>
                <strong>NIM:</strong> 1301223013
            </div>
        </div>
        <div class="identity-box">
            <div class="identity-icon">👤</div>
            <div class="identity-text">
                <strong>Nama:</strong> Yustinus Dwi Adyra<br>
                <strong>NIM:</strong> 1301223129
            </div>
        </div>
        <div class="identity-box">
            <div class="identity-icon">👤</div>
            <div class="identity-text">
                <strong>Nama:</strong> Muh. Anas Alifiano Sejati<br>
                <strong>NIM:</strong> 1301220275
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ----------------------- Judul Aplikasi ------------------------
st.markdown("<div class='center-title'>🥦 Vegetable Image Classifier 🥕</div>", unsafe_allow_html=True)
st.markdown("<div class='center-subtitle'>Upload a vegetable image and get the predicted class with confidence scores.</div>", unsafe_allow_html=True)

# ---------------------- Load Model -----------------------------
@st.cache_resource
def load_my_model():
    return load_model('baseline_model.h5')

model = load_my_model()

class_labels = [
    'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
    'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
    'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'
]

# --------------------- Preprocessing ---------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    img_array = img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------- Upload dan Prediksi ---------------------
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("❌ Gagal membuka gambar. Pastikan file gambar valid.")
        st.stop()

    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    probabilities = preds[0] * 100

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image.resize((200, 200)), caption='Uploaded Image', use_container_width=True)

    with col2:
        st.markdown(f"<h3 style='color: #FF5722;'>Predicted: {predicted_class}</h3>", unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            'Class': class_labels,
            'Probability (%)': probabilities
        }).sort_values(by='Probability (%)', ascending=False)

        # Grafik Probabilitas
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.barh(prob_df['Class'], prob_df['Probability (%)'], color='#66bb6a')
        ax.invert_yaxis()
        ax.set_xlabel('Probability (%)')
        ax.set_title('Class Probability Distribution')

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', va='center', fontsize=8)

        st.pyplot(fig)

    # --------------------- Tabel Probabilitas ---------------------
    st.markdown("### Probability Table")

    styled_df = prob_df.style.format({'Probability (%)': '{:.2f}'}).background_gradient(
        cmap='YlGn', subset=['Probability (%)']
    ).set_properties(**{
        'text-align': 'center',
        'font-size': '14px',
        'font-family': 'Segoe UI',
        'padding': '8px'
    }).set_table_styles([
        {
            'selector': 'thead th',
            'props': [
                ('background-color', '#43a047'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('border-radius', '4px'),
                ('padding', '10px')
            ]
        },
        {
            'selector': 'tbody td',
            'props': [
                ('border-bottom', '1px solid #ccc'),
                ('padding', '10px')
            ]
        },
        {
            'selector': 'tbody tr:hover',
            'props': [
                ('background-color', '#e8f5e9')
            ]
        }
    ])

    if pyarrow_available:
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("⚠️ Modul `pyarrow` tidak ditemukan. Menampilkan tabel dengan `st.table()`.")
        st.table(prob_df)