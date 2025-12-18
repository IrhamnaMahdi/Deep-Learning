import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Tingkat Stress Mahasiswa",
    page_icon="🧠",
    layout="wide"
)

# Fungsi konversi sesuai jurnal
def convert_sleep_hours(value):
    mapping = {
        '0-3 jam': 10,
        '3-6 jam': 8,
        '6-9 jam': 6,
        'Lebih dari 9 jam': 2
    }
    return mapping.get(value, 0)

def convert_study_hours(value):
    mapping = {
        'Kurang dari 1 jam': 2,
        '1-3 jam': 4,
        '3-6 jam': 6,
        'Lebih dari 6 jam': 10
    }
    return mapping.get(value, 0)

def convert_physical_activity(value):
    mapping = {
        'Tidak melakukan aktivitas fisik': 10,
        '1-2 kali': 8,
        '3-4 kali': 4,
        '5 kali': 2,
        'Lebih dari 5 kali': 2
    }
    return mapping.get(value, 0)

def convert_physical_health(value):
    mapping = {
        'Buruk': 10,
        'Kurang Baik': 8,
        'Cukup': 6,
        'Baik': 4,
        'Sangat Baik': 2
    }
    return mapping.get(value, 0)

# Load model dan scaler
@st.cache_resource
def load_models():
    try:
        model = load_model('model_stres_rmsprop_best.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, scaler, le
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Header
st.title("🧠 Sistem Prediksi Tingkat Stress Mahasiswa")
st.markdown("### Menggunakan Artificial Neural Network dengan RMSProp Optimizer")
st.markdown("---")

# Sidebar untuk informasi
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.markdown("""
    **Model:** ANN dengan RMSProp
    
    **Tingkat Stress:**
    - 🟢 **Ringan** (1-3)
    - 🟡 **Sedang** (4-6)
    - 🔴 **Berat** (7-10)
    
    **Referensi:**
    - Jurnal 1: Prediksi Tingkat Stres Pada Mahasiswa UNUGHA Cilacap
    - Jurnal 2: Klasifikasi Tingkat Stres Mahasiswa Menggunakan RMSProp untuk ANN
    """)
    
    st.markdown("---")
    st.markdown("**Developed by:** Tim Peneliti")

# Load models
model, scaler, le = load_models()

if model is None or scaler is None or le is None:
    st.error("⚠️ Model tidak dapat dimuat. Pastikan file model ada di direktori yang sama.")
    st.stop()

# Tabs untuk input
tab1, tab2 = st.tabs(["📝 Input Manual", "📊 Batch Prediction"])

with tab1:
    st.header("Input Data Mahasiswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Demografis & Kesehatan")
        
        nama = st.text_input("Nama Mahasiswa", placeholder="Masukkan nama...")
        angkatan = st.selectbox("Angkatan", [2020, 2021, 2022, 2023, 2024])
        
        sleep = st.select_slider(
            "Rata-rata Jam Tidur per Hari",
            options=['0-3 jam', '3-6 jam', '6-9 jam', 'Lebih dari 9 jam']
        )
        
        study = st.select_slider(
            "Rata-rata Jam Belajar per Minggu",
            options=['Kurang dari 1 jam', '1-3 jam', '3-6 jam', 'Lebih dari 6 jam']
        )
        
        physical = st.select_slider(
            "Aktivitas Fisik per Minggu",
            options=['Tidak melakukan aktivitas fisik', '1-2 kali', '3-4 kali', '5 kali', 'Lebih dari 5 kali']
        )
        
        health = st.select_slider(
            "Penilaian Kesehatan Fisik",
            options=['Buruk', 'Kurang Baik', 'Cukup', 'Baik', 'Sangat Baik']
        )
    
    with col2:
        st.subheader("Faktor Stress (Skala 1-5)")
        st.caption("1: Sangat Tidak Setuju | 5: Sangat Setuju")
        
        q1 = st.slider("Terbebani harapan orang tua", 1, 5, 3)
        q2 = st.slider("Terganggu fokus karena ajakan teman", 1, 5, 3)
        q3 = st.slider("Kesulitan menjaga pola tidur", 1, 5, 3)
        q4 = st.slider("Kondisi keuangan menjadi kendala", 1, 5, 3)
        q5 = st.slider("Kesulitan memahami materi dosen", 1, 5, 3)
        q6 = st.slider("Kesulitan menemukan literatur", 1, 5, 3)
        q7 = st.slider("Waktu istirahat terbatas", 1, 5, 3)
        q8 = st.slider("Lingkungan tidak kondusif", 1, 5, 3)
    
    st.markdown("---")
    
    if st.button("🔮 Prediksi Tingkat Stress", type="primary", use_container_width=True):
        if not nama:
            st.warning("⚠️ Mohon masukkan nama mahasiswa")
        else:
            with st.spinner("Melakukan prediksi..."):
                # Konversi input
                features = np.array([[
                    convert_sleep_hours(sleep),
                    convert_study_hours(study),
                    convert_physical_activity(physical),
                    convert_physical_health(health),
                    q1, q2, q3, q4, q5, q6, q7, q8
                ]])
                
                # Scaling
                features_scaled = scaler.transform(features)
                
                # Prediksi
                prediction = model.predict(features_scaled, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = prediction[0][predicted_class] * 100
                
                # Decode label
                stress_level = le.classes_[predicted_class]
                
                # Display hasil
                st.success("✅ Prediksi Berhasil!")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    if stress_level == "Ringan":
                        st.metric("Tingkat Stress", "🟢 Ringan", delta="Normal")
                    elif stress_level == "Sedang":
                        st.metric("Tingkat Stress", "🟡 Sedang", delta="Perhatian")
                    else:
                        st.metric("Tingkat Stress", "🔴 Berat", delta="Waspada", delta_color="inverse")
                
                with result_col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                with result_col3:
                    st.metric("Nama", nama)
                
                # Visualisasi probabilitas
                st.markdown("### Distribusi Probabilitas")
                prob_df = pd.DataFrame({
                    'Kategori': le.classes_,
                    'Probabilitas (%)': prediction[0] * 100
                })
                st.bar_chart(prob_df.set_index('Kategori'))
                
                # Rekomendasi
                st.markdown("### 💡 Rekomendasi")
                if stress_level == "Ringan":
                    st.info("""
                    **Tingkat stress Anda masih dalam batas normal.** 
                    - Pertahankan pola hidup sehat
                    - Tetap jaga keseimbangan belajar dan istirahat
                    - Lakukan aktivitas yang menyenangkan
                    """)
                elif stress_level == "Sedang":
                    st.warning("""
                    **Tingkat stress Anda mulai meningkat.**
                    - Atur manajemen waktu dengan lebih baik
                    - Tingkatkan kualitas tidur
                    - Lakukan relaksasi atau olahraga ringan
                    - Komunikasikan masalah dengan orang terdekat
                    """)
                else:
                    st.error("""
                    **Tingkat stress Anda cukup tinggi.**
                    - Segera konsultasi dengan psikolog atau konselor
                    - Evaluasi beban akademik dan prioritas
                    - Perbaiki pola tidur dan makan
                    - Lakukan aktivitas stress relief (meditasi, olahraga)
                    - Hubungi: Layanan Konseling Kampus
                    """)

with tab2:
    st.header("Prediksi Batch")
    st.markdown("Upload file CSV dengan format sesuai template")
    
    # Template download
    template_data = {
        'Nama': ['Contoh 1', 'Contoh 2'],
        'Angkatan': [2022, 2023],
        'Rata-rata Jam Tidur per Hari': ['3-6 jam', '6-9 jam'],
        'Rata-rata Jam Belajar per Minggu': ['1-3 jam', '3-6 jam'],
        'Rata-rata Melakukan Aktivitas Fisik per Minggu': ['1-2 kali', '3-4 kali'],
        'Penilaian Kesehatan Fisik Secara Umum': ['Cukup', 'Baik'],
        'q1': [3, 2],
        'q2': [3, 2],
        'q3': [5, 3],
        'q4': [2, 4],
        'q5': [1, 4],
        'q6': [1, 1],
        'q7': [1, 3],
        'q8': [3, 3]
    }
    template_df = pd.DataFrame(template_data)
    
    st.download_button(
        label="📥 Download Template CSV",
        data=template_df.to_csv(index=False),
        file_name="template_prediksi_stress.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview Data:", df.head())
            
            if st.button("🚀 Mulai Prediksi Batch"):
                with st.spinner("Processing..."):
                    # Konversi fitur
                    df_processed = df.copy()
                    df_processed['x1_waktutidur'] = df['Rata-rata Jam Tidur per Hari'].apply(convert_sleep_hours)
                    df_processed['x2_kebiasaanstudi'] = df['Rata-rata Jam Belajar per Minggu'].apply(convert_study_hours)
                    df_processed['x3_aktivitasfisik'] = df['Rata-rata Melakukan Aktivitas Fisik per Minggu'].apply(convert_physical_activity)
                    df_processed['x4_kesehatanfisik'] = df['Penilaian Kesehatan Fisik Secara Umum'].apply(convert_physical_health)
                    
                    # Buat feature array
                    feature_cols = ['x1_waktutidur', 'x2_kebiasaanstudi', 'x3_aktivitasfisik', 'x4_kesehatanfisik',
                                   'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8']
                    X_batch = df_processed[feature_cols].values
                    
                    # Scaling & Prediksi
                    X_batch_scaled = scaler.transform(X_batch)
                    predictions = model.predict(X_batch_scaled, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)
                    
                    # Tambahkan hasil ke dataframe
                    df['Prediksi_Stress'] = le.inverse_transform(predicted_classes)
                    df['Confidence'] = [predictions[i][predicted_classes[i]] * 100 for i in range(len(predictions))]
                    
                    st.success("✅ Prediksi selesai!")
                    st.dataframe(df)
                    
                    # Summary
                    st.markdown("### 📊 Ringkasan Hasil")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        ringan_count = (df['Prediksi_Stress'] == 'Ringan').sum()
                        st.metric("🟢 Ringan", ringan_count)
                    
                    with col2:
                        sedang_count = (df['Prediksi_Stress'] == 'Sedang').sum()
                        st.metric("🟡 Sedang", sedang_count)
                    
                    with col3:
                        berat_count = (df['Prediksi_Stress'] == 'Berat').sum()
                        st.metric("🔴 Berat", berat_count)
                    
                    # Download hasil
                    st.download_button(
                        label="📥 Download Hasil Prediksi",
                        data=df.to_csv(index=False),
                        file_name="hasil_prediksi_stress.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>© 2024 Sistem Prediksi Tingkat Stress Mahasiswa | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)