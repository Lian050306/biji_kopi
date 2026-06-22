import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

st.set_page_config(page_title="Identifikasi Mutu Biji Kopi", layout="wide")

st.title("Sistem Identifikasi Mutu Biji Kopi")
st.subheader("Standarisasi Kualitas Ekspor dengan MobileNetV2")

@st.cache_resource
def load_model_cached():
    model = load_model('model_final.keras', compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model_cached()

class_names = ['specialty_brazil', 'premium_grade_1', 'asalan', 'rusak']
class_info = {
    'specialty_brazil': {'label': 'Specialty Brazil', 'rekomendasi': 'Layak ekspor premium', 'layak_ekspor': True},
    'premium_grade_1': {'label': 'Premium Grade 1', 'rekomendasi': 'Layak ekspor standar', 'layak_ekspor': True},
    'asalan': {'label': 'Asalan', 'rekomendasi': 'Perlu pemrosesan ulang', 'layak_ekspor': False},
    'rusak': {'label': 'Rusak', 'rekomendasi': 'Harus disortir', 'layak_ekspor': False}
}

uploaded_file = st.file_uploader("Upload gambar biji kopi", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Gambar yang diupload', use_container_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner('Menganalisis gambar...'):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100
    
    with col2:
        st.subheader("Hasil Identifikasi")
        kelas = class_names[predicted_class]
        info = class_info[kelas]
        
        st.markdown(f"**Kelas Terdeteksi:** {info['label']}")
        st.markdown(f"**Akurasi:** {confidence:.2f}%")
        st.markdown(f"**Rekomendasi:** {info['rekomendasi']}")
        
        if info['layak_ekspor']:
            st.success("LAYAK EKSPOR")
        else:
            st.error("TIDAK LAYAK EKSPOR")
    
    st.subheader("Detail Prediksi")
    for i, kelas in enumerate(class_names):
        persentase = float(predictions[0][i] * 100)
        st.progress(persentase/100)
        st.write(f"{class_info[kelas]['label']}: {persentase:.2f}%")