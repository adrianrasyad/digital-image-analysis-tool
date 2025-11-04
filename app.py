import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from skimage.metrics import structural_similarity as ssim

# --- KONFIGURASI STREAMLIT ---
st.set_page_config(layout="wide", page_title="DIP - Analisis Statistik & Pencocokan Citra")

# --- FUNGSI UTILITY ---

def load_image_and_process(uploaded_file):
    """Membaca file yang diunggah dan memastikannya valid (grayscale)."""
    if uploaded_file is None:
        return None
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Baca sebagai citra Grayscale
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None:
            st.error("Gagal memuat citra. Pastikan format file didukung.")
            return None
        
        # Resize citra jika terlalu besar, untuk performa
        if img_gray.shape[0] > 512 or img_gray.shape[1] > 512:
            img_gray = cv2.resize(img_gray, (512, 512), interpolation=cv2.INTER_AREA)

        return img_gray
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses citra: {e}")
        return None

# --- MODUL 1: ANALISIS STATISTIK CITRA TUNGGAL ---

def calculate_stats(img):
    """Menghitung metrik statistik dari citra (array NumPy)."""
    
    # Normalisasi data ke skala 0-1 untuk statistik yang lebih stabil
    data = img.flatten().astype(np.float64)
    
    # Metrik Dasar
    mean_val = np.mean(img)
    std_dev_val = np.std(img)

    # 1. Pearson Correlation (Korelasi antara piksel dan tetangga terdekat)
    # Kita akan menggunakan korelasi antara baris dan kolom yang berdekatan.
    corr_x = np.corrcoef(img[:, :-1].flatten(), img[:, 1:].flatten())[0, 1]
    corr_y = np.corrcoef(img[:-1, :].flatten(), img[1:, :].flatten())[0, 1]
    pearson_corr = (corr_x + corr_y) / 2
    
    # 2. Skewness
    img_skew = skew(data)
    
    # 3. Kurtosis
    img_kurtosis = kurtosis(data) # Fisher's definition (normal = 0)
    
    # 4. Entropy (untuk mengukur tingkat informasi/ketidakpastian)
    hist, _ = np.histogram(data, bins=256, range=[0, 256], density=True)
    hist = hist[hist > 0] # Hanya gunakan bin yang memiliki nilai > 0
    entropy = -np.sum(hist * np.log2(hist))
    
    # 5. Chi-Square (vs Uniform Hist)
    # Histogram rata ideal
    ideal_hist = np.ones(256) * (len(data) / 256)
    
    # Hitung histogram citra
    img_hist, _ = np.histogram(data, bins=256, range=[0, 256])
    
    # Chi-Square Test (Membandingkan dengan distribusi seragam)
    chi_square_val = np.sum(((img_hist - ideal_hist)**2) / ideal_hist)

    return {
        "Mean (Rata-Rata Kecerahan)": mean_val,
        "Standard Deviation (Kontras)": std_dev_val,
        "Pearson Correlation (Rata-rata):": pearson_corr,
        "Skewness (Kemiringan Distribusi):": img_skew,
        "Kurtosis (Ketajaman Puncak Distribusi):": img_kurtosis,
        "Entropy (Tingkat Informasi):": entropy,
        "Chi-Square (vs Uniform Hist):": chi_square_val
    }

def get_entropy_interpretation(entropy):
    """Memberikan interpretasi verbal berdasarkan nilai Entropy."""
    if entropy > 7.0:
        return "Tinggi: Citra memiliki kontras yang baik dan detail yang kaya. Distribusi piksel cenderung merata (ideal)."
    elif entropy > 5.5:
        return "Sedang: Citra memiliki informasi yang cukup, namun mungkin sedikit gelap/terang atau kontrasnya agak kurang."
    else:
        return "Rendah: Citra mungkin didominasi oleh warna yang seragam (kontras rendah) atau memiliki masalah *quantization*."

def get_skewness_interpretation(skew_val):
    """Memberikan interpretasi verbal berdasarkan nilai Skewness."""
    if skew_val > 0.5:
        return "Positif (Miring Kanan): Histogram dominan berada di sisi gelap (intensitas rendah). Citra cenderung gelap."
    elif skew_val < -0.5:
        return "Negatif (Miring Kiri): Histogram dominan berada di sisi terang (intensitas tinggi). Citra cenderung terang/putih."
    else:
        return "Netral: Distribusi piksel relatif seimbang di tengah (kecerahan moderat)."

def get_kurtosis_interpretation(kurt_val):
    """Memberikan interpretasi verbal berdasarkan nilai Kurtosis."""
    if kurt_val > 1.0:
        return "Leptokurtic (Puncak Tajam): Distribusi sangat terpusat (piksel terkonsentrasi pada beberapa nilai). Kontras citra rendah."
    elif kurt_val < -1.0:
        return "Platykurtic (Puncak Datar): Distribusi sangat menyebar. Piksel tersebar merata. Kontras citra sangat tinggi."
    else:
        return "Mesokurtic: Distribusi normal. Citra memiliki kontras dan kecerahan yang seimbang."


def module_statistical_analysis():
    st.header("🔬 Analisis Statistik Citra Tunggal")
    st.caption("Hitung metrik statistik berdasarkan distribusi piksel citra yang diunggah.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Unggah Citra Pertama", type=["jpg", "jpeg", "png"], key="stat_upload")

    img_gray = load_image_and_process(uploaded_file)
    
    if img_gray is not None:
        col_img, col_stats = st.columns([1, 1])

        with col_img:
            st.image(img_gray, caption="Citra Asli (Grayscale)", use_container_width=True)
            
            # Tambahkan bagian Histogram di sini agar lebih dekat dengan interpretasi Skewness/Kurtosis
            st.markdown("### Histogram Intensitas Piksel")
            fig, ax = plt.subplots()
            ax.hist(img_gray.flatten(), bins=256, range=[0, 256], color='gray', alpha=0.7)
            ax.set_title("Distribusi Piksel")
            ax.set_xlabel("Intensitas Piksel (0=Gelap, 255=Terang)")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)
            plt.close(fig)

        with col_stats:
            st.markdown("### Hasil Statistik dan Interpretasi Citra")
            stats = calculate_stats(img_gray)
            
            # Pisahkan Rata-Rata dan Standar Deviasi di baris pertama
            col_mean, col_std = st.columns(2)
            with col_mean:
                st.metric(label="1. Mean (Rata-Rata Kecerahan)", value=f"{stats['Mean (Rata-Rata Kecerahan)']:.2f}")
                st.markdown(f"**Interpretasi:** Nilai ini menunjukkan kecerahan rata-rata citra (midpoint ideal = 127).")
            with col_std:
                st.metric(label="2. Standard Deviation (Kontras)", value=f"{stats['Standard Deviation (Kontras)']:.2f}")
                st.markdown(f"**Interpretasi:** Semakin tinggi, semakin besar rentang kontras dalam citra.")
            
            st.markdown("---")

            # Skewness
            st.metric(label="3. Skewness (Kemiringan Distribusi)", value=f"{stats['Skewness (Kemiringan Distribusi):']:.4f}")
            st.markdown(f"**Interpretasi Skewness:** {get_skewness_interpretation(stats['Skewness (Kemiringan Distribusi):'])}")
            
            # Kurtosis
            st.metric(label="4. Kurtosis (Ketajaman Puncak Distribusi)", value=f"{stats['Kurtosis (Ketajaman Puncak Distribusi):']:.4f}")
            st.markdown(f"**Interpretasi Kurtosis:** {get_kurtosis_interpretation(stats['Kurtosis (Ketajaman Puncak Distribusi):'])}")
            
            # Entropy
            st.metric(label="5. Entropy (Tingkat Informasi)", value=f"{stats['Entropy (Tingkat Informasi):']:.4f}")
            st.markdown(f"**Interpretasi Entropy:** {get_entropy_interpretation(stats['Entropy (Tingkat Informasi):'])}")
            
            # Metrik Lanjutan
            st.markdown("---")
            st.markdown("##### Metrik Lanjutan (Untuk Analisis Detail)")
            
            st.metric(label="6. Pearson Correlation (Rata-rata)", value=f"{stats['Pearson Correlation (Rata-rata):']:.4f}", help="Mengukur seberapa erat korelasi piksel tetangga (semakin dekat ke 1.0, semakin halus citra).")
            st.metric(label="7. Chi-Square (vs Uniform Hist)", value=f"{stats['Chi-Square (vs Uniform Hist):']:.4f}", help="Mengukur perbedaan histogram citra dari histogram ideal (merata). Nilai yang lebih kecil mendekati nol, menunjukkan distribusi piksel lebih merata.")


# --- MODUL 2: PENCocokan CITRA (IMAGE MATCHING) ---

def compare_images(img1, img2):
    """Melakukan perbandingan citra menggunakan beberapa metrik dan mengembalikan citra perbedaan."""
    
    # 1. Resize citra kedua agar dimensinya sama dengan citra pertama (Kritis untuk SSIM dan Hist)
    h1, w1 = img1.shape
    img2_resized = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)

    # a. Structural Similarity Index (SSIM) - Mengukur kemiripan persepsi struktural
    # SSIM membutuhkan tipe data float
    (s, diff) = ssim(img1, img2_resized, full=True, data_range=img1.max() - img1.min())
    
    # Untuk visualisasi perbedaan, kita konversi array diff (float, 0-1) ke 8-bit integer (0-255)
    # 1.0 - diff digunakan karena SSIM diff map menunjukkan kesamaan, kita ingin menunjukkan perbedaan
    diff = (1.0 - diff) * 255
    diff = diff.astype("uint8")
    
    # b. Histogram Comparison (Chi-Square Distance) - Mengukur kemiripan distribusi warna/intensitas
    # Hitung histogram
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256], accumulate=False)
    hist2 = cv2.calcHist([img2_resized], [0], None, [256], [0, 256], accumulate=False)
    
    # Normalisasi Histogram untuk perbandingan yang lebih baik
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Hitung perbandingan menggunakan metode Chi-Square (semakin kecil, semakin mirip)
    # CV_COMP_CHISQR: Semakin kecil nilainya, semakin mirip
    chi_square_hist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR) 
    
    # c. Mean Squared Error (MSE) - Mengukur perbedaan rata-rata kuadrat (semakin kecil, semakin mirip)
    mse = np.sum((img1.astype("float") - img2_resized.astype("float")) ** 2)
    mse /= float(w1 * h1)

    # Mengembalikan SSIM, Chi-Square, MSE, dan citra perbedaan (diff image)
    return s, chi_square_hist, mse, diff

def get_similarity_status(ssim_val):
    """Memberikan status verbal berdasarkan nilai SSIM."""
    if ssim_val >= 0.95:
        return "💚 Sangat Mirip / Identik"
    elif ssim_val >= 0.85:
        return "💛 Mirip (Perbedaan kecil)"
    elif ssim_val >= 0.70:
        return "🧡 Cukup Mirip (Mirip di struktur, beda di warna/kontras)"
    elif ssim_val >= 0.50:
        return "💔 Agak Berbeda (Hanya sebagian struktur yang sama)"
    else:
        return "🖤 Jauh Berbeda (Struktur citra tidak mirip)"


def module_image_matching():
    st.header("🖼️ Pencocokan Citra (Image Matching)")
    st.caption("Bandingkan dua citra yang tidak sama menggunakan metrik kemiripan.")
    st.markdown("---")

    col_upload1, col_upload2 = st.columns(2)

    with col_upload1:
        uploaded_file1 = st.file_uploader("Unggah Citra Pertama (Target)", type=["jpg", "jpeg", "png"], key="match_upload1")
        img1 = load_image_and_process(uploaded_file1)
        # Pengecekan ditambahkan di sini:
        if img1 is not None:
            st.image(img1, caption="Citra 1", use_container_width=True)

    with col_upload2:
        uploaded_file2 = st.file_uploader("Unggah Citra Kedua (Query)", type=["jpg", "jpeg", "png"], key="match_upload2")
        img2 = load_image_and_process(uploaded_file2)
        # Pengecekan ditambahkan di sini:
        if img2 is not None:
            st.image(img2, caption="Citra 2 (akan di-resize)", use_container_width=True)

    st.markdown("---")

    if img1 is not None and img2 is not None:
        st.subheader("Hasil Pencocokan (Matching)")
        
        # Panggil fungsi perbandingan
        ssim_val, chi_square_hist, mse_val, diff_image = compare_images(img1, img2)
        
        # 1. Status Kemiripan Tekstual
        similarity_status = get_similarity_status(ssim_val)
        st.info(f"**STATUS KEMIRIPAN (Berdasarkan SSIM):** {similarity_status}")

        # PENTING: Mendefinisikan kolom metrik HANYA di dalam blok kondisional.
        col_metrics = st.columns(3)
        
        with col_metrics[0]:
            st.metric(
                label="Structural Similarity Index (SSIM)", 
                value=f"{ssim_val:.4f}",
                help="Mengukur kemiripan struktural (skala 0 hingga 1.0). Nilai 1.0 berarti citra identik."
            )
            
        with col_metrics[1]:
            st.metric(
                label="Histogram Chi-Square Distance", 
                value=f"{chi_square_hist:.2f}",
                help="Mengukur perbedaan distribusi piksel. Nilai 0 berarti distribusi identik."
            )
            
        with col_metrics[2]:
            st.metric(
                label="Mean Squared Error (MSE)", 
                value=f"{mse_val:.2f}",
                help="Mengukur perbedaan rata-rata kuadrat piksel. Semakin kecil, semakin mirip."
            )
        
        st.markdown("---")
        
        # 2. Visualisasi Perbedaan
        st.subheader("Visualisasi Perbedaan (Difference Map)")
        st.caption("Area yang lebih terang/putih menunjukkan perbedaan yang lebih besar antara Citra 1 dan Citra 2.")
        
        # Difference map
        st.image(diff_image, caption="Peta Perbedaan Piksel", use_container_width=True)


# --- FUNGSI UTAMA (MAIN APP) ---

def main():
    st.sidebar.title("Pilih Modul")
    module_selection = st.sidebar.selectbox(
        "Pilih Tugas:",
        ("1. Analisis Statistik Citra Tunggal", "2. Pencocokan Citra (Image Matching)")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Perhatian:** Modul ini membutuhkan citra yang diunggah harus dalam format Grayscale.")

    if module_selection == "1. Analisis Statistik Citra Tunggal":
        module_statistical_analysis()
    elif module_selection == "2. Pencocokan Citra (Image Matching)":
        module_image_matching()

if __name__ == "__main__":
    main()
