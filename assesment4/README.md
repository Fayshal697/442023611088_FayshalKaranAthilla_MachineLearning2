# 🍎🍌 Klasifikasi Gambar: Apel vs Pisang dengan Transfer Learning

Proyek ini menggunakan **Transfer Learning** dengan arsitektur **MobileNetV2** untuk mengklasifikasikan gambar ke dalam dua kategori: **apel** dan **pisang**. Kumpulan data terdiri dari gambar apel dan pisang segar dan busuk, yang telah diberi label ulang ke dalam dua kelas untuk penyederhanaan.

## 📁 Ringkasan Kumpulan Data

- **Sumber**: Kumpulan data kustom (dari 6 kelas asli)
- **Kelas yang Digunakan**:
- `freshapples`, `rottenapples` → **apple**
- `freshbanana`, `rottenbanana` → **banana**
- **Total Gambar**:
- Pelatihan: 868
- Validasi: 216

## ⚙️ Arsitektur Model

- Model Dasar: `MobileNetV2` (sudah dilatih, tanpa lapisan atas)
- Lapisan Tambahan:
- `GlobalAveragePooling2D`
- `Dense(2, activation='softmax')`
- Pengoptimal: `Adam`
- Kerugian: `categorical_crossentropy`
- Metrik: `accuracy`

## 📊 Hasil Pelatihan

- Epoch: 10
- Model mencapai hampir **100% akurasi** pada set pelatihan dan validasi.
- Namun, matriks kebingungan mengungkap kesalahan klasifikasi yang seimbang antara kedua kelas.

### Akurasi & Kerugian

![Akurasi dan Loss](akurasiloss.png)

### Matriks Kebingungan

![Confusion Matrix](confmatrix.png)

## 🔍 Ringkasan Evaluasi

| Metrik | Nilai |
|----------------|---------------|
| Akurasi Latih | ~100% |
| Akurasi Val | ~100% |
| Kesalahan Klasifikasi | 51 untuk setiap kelas |

**Catatan**: Meskipun akurasinya tinggi, kebingungan yang seimbang menunjukkan potensi bias model atau kesamaan kelas dalam fitur visual.
