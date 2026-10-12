# 👥 Prediksi Karyawan yang Berisiko Keluar (Employee Attrition)

Proyek *machine learning* untuk memperkirakan apakah seorang karyawan akan **keluar dari perusahaan** (`Attrition = Yes`) atau **bertahan** (`Attrition = No`). Hasilnya membantu tim HR memantau karyawan yang berisiko lebih dulu.

Seluruh proses (eksplorasi data → pemodelan → evaluasi → inferensi data baru) ada di satu notebook: **`prediksi_karyawan_keluar_8.ipynb`**.

---

## 📌 Ringkasan Proyek

| Item | Keterangan |
|---|---|
| Tugas | Klasifikasi biner (`Attrition`: Yes = 1, No = 0) |
| Dataset | IBM HR Analytics Employee Attrition (`WA_Fn-UseC_-HR-Employee-Attrition_1.csv`) |
| Ukuran data | 1.470 karyawan, 35 kolom |
| Proporsi target | ±16% karyawan keluar (data tidak seimbang) |
| Model dibandingkan | Logistic Regression dan Random Forest |
| Model terpilih | **Logistic Regression** (PR-AUC cross-validation tertinggi) |
| Pemisahan data | 80% latih (1.176 baris) / 20% uji (294 baris), `stratify=y` |
| Fitur akhir | 15 dari 30 fitur |

---

## 🧭 Prinsip yang Dipegang

| Prinsip | Penjelasan |
|---|---|
| Split lebih dulu | Data dipisah sebelum belajar apa pun agar penilaian jujur (tanpa *data leakage*) |
| Pakai `Pipeline` | Scaler dan encoder hanya belajar dari data latih |
| Pilih model dengan *cross-validation* | Data uji hanya dipakai untuk penilaian akhir |
| Metrik yang tepat | Recall, Precision, F1, ROC-AUC, dan PR-AUC, bukan akurasi saja |
| Tangani data tidak seimbang | `class_weight` dan penyetelan *threshold* |

---

## 🔄 Alur Notebook

1. **Import library** dengan `RANDOM_STATE = 42` agar hasil dapat direproduksi.
2. **Load data** (pemisah dan encoding terdeteksi otomatis: `sep=None`, `encoding="utf-8-sig"`).
3. **Eksplorasi data (EDA)**: cek nilai kosong, duplikat, kolom konstan, dan keseimbangan target.
4. **Pembersihan data**: buang kolom konstan (`EmployeeCount`, `Over18`, `StandardHours`) dan ID (`EmployeeNumber`), ubah target menjadi 0/1.
5. **Split train/test** (sebelum preprocessing).
   - 5.1 Analisis korelasi antar fitur (Pearson dan Cramér's V), hanya dari data latih.
   - 5.2 Seleksi fitur: buang fitur redundan, fitur yang dinolkan L1 (Lasso), dan fitur numerik yang tidak menambah PR-AUC (*backward elimination*).
6. **Preprocessing dengan Pipeline**: `StandardScaler` (numerik) dan `OneHotEncoder` (kategori).
7. **Bandingkan model** dengan *Stratified 5-fold cross-validation*.
8. **Atur threshold** berdasarkan F1 terbaik dari prediksi *out-of-fold* pada data latih.
9. **Ujian akhir** di data uji, lengkap dengan classification report, confusion matrix, kurva ROC dan Precision-Recall.
10. **Faktor berpengaruh** menggunakan *Permutation Importance*.
11. **Simpan model** (pipeline + threshold + daftar kolom) ke `model_attrition.joblib`.
12. **Inferensi data baru**: validasi data, prediksi batch, evaluasi opsional, simpan hasil, uji satu karyawan secara manual, dan input data karyawan secara interaktif lewat `input()`.

---

## 🧹 Seleksi Fitur: 30 → 15

**Fitur yang dipakai (15):**
`Age`, `BusinessTravel`, `DistanceFromHome`, `EducationField`, `EnvironmentSatisfaction`, `Gender`, `JobInvolvement`, `JobRole`, `JobSatisfaction`, `MaritalStatus`, `NumCompaniesWorked`, `OverTime`, `TrainingTimesLastYear`, `WorkLifeBalance`, `YearsWithCurrManager`

**Fitur yang dibuang (15):**

| Alasan | Fitur |
|---|---|
| Redundan (berkorelasi kuat) | `MonthlyIncome`, `JobLevel`, `YearsAtCompany`, `YearsInCurrentRole`, `PerformanceRating`, `Department` |
| Dinolkan L1 (tidak berguna) | `Education`, `HourlyRate`, `MonthlyRate` |
| Tidak menambah performa | `PercentSalaryHike`, `DailyRate`, `TotalWorkingYears`, `StockOptionLevel`, `YearsSinceLastPromotion`, `RelationshipSatisfaction` |

> Fitur yang dibuang bukan berarti tidak berhubungan dengan `Attrition`. Informasinya sudah terwakili fitur lain. Seluruh seleksi hanya memakai data latih.

---

## 📊 Hasil

### Perbandingan model (cross-validation 5-fold, data latih)

| Model | Akurasi | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0,756 | 0,829 | **0,629** | 0,726 | 0,371 | 0,490 |
| Random Forest | 0,867 | 0,801 | 0,570 | 0,389 | 0,665 | 0,478 |

Model dipilih berdasarkan **PR-AUC** karena paling peka terhadap kemampuan menemukan kelas yang langka.

### Evaluasi akhir (data uji, 294 karyawan, 47 yang keluar)

| Cara | Akurasi | Recall | Precision | F1 |
|---|---|---|---|---|
| Tebak semua "No" | 0,840 | 0,000 | 0,000 | 0,000 |
| Model, threshold 0,5 | 0,793 | 0,745 | 0,417 | 0,534 |
| Model, threshold 0,70 (terpilih) | 0,847 | 0,404 | 0,528 | 0,458 |

- **ROC-AUC data uji:** 0,807
- **PR-AUC data uji:** 0,550

### Cara membaca hasil

- Menebak "semua bertahan" sudah akurat 84%, tetapi tidak menemukan satu pun karyawan yang keluar. Karena itu **akurasi tidak boleh berdiri sendiri**.
- Threshold 0,70 menghasilkan Precision lebih tinggi (lebih sedikit salah alarm), sedangkan threshold 0,5 menangkap lebih banyak karyawan yang akan keluar (Recall 0,745). **Threshold dapat disesuaikan dengan kebutuhan HR**: jika biaya kehilangan karyawan besar, pilih threshold lebih rendah.
- Jumlah karyawan yang keluar di data uji hanya 47, sehingga angka metrik mudah berubah. Jangan menarik kesimpulan yang terlalu kuat.
- Faktor yang ditunjukkan model adalah **keterkaitan, bukan sebab-akibat**. Gunakan sebagai petunjuk awal untuk diskusi dengan tim HR.

---

## 🛠️ Instalasi

Kebutuhan: Python 3.9+ dan library berikut.

```bash
pip install numpy pandas matplotlib scipy scikit-learn joblib jupyter
```

## ▶️ Cara Menjalankan

1. Siapkan file dataset **`WA_Fn-UseC_-HR-Employee-Attrition_1.csv`** dan letakkan satu folder dengan notebook (atau ubah variabel `DATA_PATH` di bagian 2).
2. Buka notebook:
   ```bash
   jupyter notebook prediksi_karyawan_keluar_8.ipynb
   ```
3. Jalankan semua sel secara berurutan (*Run All*).

> 💡 **Google Colab:** hapus tanda `#` pada baris `files.upload()` untuk mengunggah dataset, atau unggah lewat panel *Files*.

---

## 🔮 Menggunakan Model pada Data Baru

Bagian 12 notebook **berdiri sendiri**: hanya butuh `model_attrition.joblib` dan file CSV data baru, tanpa melatih ulang model.

1. Siapkan CSV (misalnya `data_karyawan_baru.csv`) dengan nama kolom sama seperti data asli. Cukup 15 kolom yang dibutuhkan model, kolom lain boleh ada.
2. Kolom `Attrition` **opsional**. Jika ada, hasil akan dievaluasi. Jika tidak ada, hanya prediksi yang dihasilkan.
3. Ubah `DATA_BARU_PATH`, lalu jalankan bagian 12.
4. Jika file tidak ditemukan, notebook membuat `template_data_baru.csv` dan menjalankan **mode demo** (baris contoh berasal dari data uji, jadi belum benar-benar "baru").

**Validasi otomatis:** kolom wajib yang kurang, nilai kosong, dan kolom angka berisi teks akan menghentikan proses dengan pesan jelas. Angka di luar rentang data latih atau kategori baru hanya diberi peringatan.

**Keluaran** (`hasil_prediksi_attrition.csv`, diurutkan dari yang paling berisiko):

| Kolom | Arti |
|---|---|
| `Peluang_Keluar_%` | Peluang karyawan keluar |
| `Prediksi` | Keputusan berdasarkan threshold (mis. "Berisiko keluar") |
| `Tingkat_Risiko` | **Tinggi**: di atas threshold. **Sedang**: antara setengah threshold dan threshold (perlu dipantau). **Rendah**: di bawahnya |

Untuk satu karyawan, ada dua cara:

- **Lewat dictionary:** `prediksi_attrition(dict_data)`. Semua kolom yang dibutuhkan model harus ada di dictionary.
- **Lewat input interaktif (bagian 12.7):** jalankan `prediksi_interaktif()`. Notebook menanyakan 15 kolom satu per satu, lalu menampilkan peluang keluar, keputusan, dan tingkat risiko. Setelah itu notebook menawarkan untuk memprediksi karyawan lain.

```python
data_input, hasil_input = prediksi_interaktif()
```

Beberapa hal yang dilakukan otomatis oleh input interaktif:

| Fitur | Penjelasan |
|---|---|
| Daftar kolom dinamis | Diambil dari `model_attrition.joblib`, jadi tetap benar jika seleksi fitur berubah |
| Menu untuk kolom kategori | Pilih dengan nomor atau ketik teksnya (mis. `OverTime`, `JobRole`) |
| Validasi angka | Harus angka, bulat jika memang bulat, tidak negatif; kolom skala (mis. `JobSatisfaction`) dijaga di 1-4 |
| Rentang data latih | Ditampilkan sebagai panduan. Angka di luar rentang tetap diterima, tetapi muncul peringatan saat prediksi |

Fungsi `input_karyawan()` juga dapat dipakai sendiri untuk mengumpulkan data satu karyawan, misalnya `prediksi_attrition(input_karyawan())`.

---

## 📁 Struktur File

```
.
├── prediksi_karyawan_keluar_8.ipynb        # notebook utama
├── WA_Fn-UseC_-HR-Employee-Attrition_1.csv # dataset (siapkan sendiri)
├── model_attrition.joblib                  # model tersimpan (dibuat notebook)
├── template_data_baru.csv                  # template data baru (dibuat notebook, mode demo)
├── data_karyawan_baru.csv                  # data karyawan baru (opsional, dari Anda)
├── hasil_prediksi_attrition.csv            # hasil prediksi (dibuat notebook)
└── README.md
```

---

## ⚠️ Keterbatasan

- Dataset berukuran kecil dan bersifat sintetis/contoh, sehingga performa pada data perusahaan nyata dapat berbeda.
- Recall pada threshold terpilih (0,40) berarti sebagian karyawan yang akan keluar tetap terlewat.
- Hasil seleksi fitur pada *cross-validation* agak optimistis karena dipilih dari data latih yang sama. Penilaian yang jujur ada pada evaluasi di data uji.
- Model perlu dipantau dan dilatih ulang secara berkala jika kondisi perusahaan berubah.
- Gunakan prediksi sebagai **alat bantu keputusan**, bukan satu-satunya dasar keputusan terhadap karyawan.
