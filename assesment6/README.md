# EKSPLORASI AUTOENCODER: PEMODELAN REPRESENTASI LATEN WAJAH MENGGUNAKAN VARIATIONAL AUTOENCODER (VAE) PADA DATASET SELEBRITIS HOLLYWOOD

Proyek ini mengimplementasikan Variational Autoencoder (VAE) untuk merekonstruksi gambar wajah manusia menggunakan dataset CelebA. Tujuan utamanya adalah mempelajari representasi laten yang ringkas dari fitur wajah serta mengevaluasi kualitas rekonstruksi dan distribusi ruang laten.

---

## 📦 Fitur Utama

- Model VAE yang dibangun menggunakan PyTorch
- Pelatihan menggunakan dataset CelebA dengan preprocessing gambar
- Rekonstruksi gambar wajah
- Visualisasi ruang laten menggunakan PCA
- Pelacakan dan plotting loss selama pelatihan

---

## 🧠 Arsitektur Model

Model VAE terdiri dari:

- **Encoder**: Layer konvolusional → layer linear → menghasilkan `mu` dan `logvar`
- **Trik Reparameterisasi**: Sampling dari ruang laten menggunakan `mu` dan `logvar`
- **Decoder**: Layer linear → layer transposed convolution untuk merekonstruksi gambar

---

## 📊 Hasil & Visualisasi

### 🔻 Grafik Loss
Menampilkan penurunan loss yang konsisten selama proses pelatihan.

![Grafik Loss](/assesment6/graphic_loss.png)

### 🎭 Rekonstruksi
Perbandingan antara gambar asli dan hasil rekonstruksi.

![Asli vs Rekonstruksi](/assesment6/Original_vs_Reconstructured.png)

### 🌀 Distribusi Ruang Laten (PCA)
Visualisasi distribusi vektor laten dalam ruang 2 dimensi menggunakan PCA.

![Ruang Laten](/assesment6/Space_Latent_After_PCA.png)
