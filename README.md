# 🏥 Oral Cancer Detection Telegram Bot

Bot Telegram berbasis AI untuk deteksi dini kanker mulut menggunakan **MobileNetV2** dan **Transfer Learning**-nya Gigital

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Cara Kerja](#-cara-kerja)
- [Instalasi](#-instalasi)
- [Training Model](#-training-model)
- [Menjalankan Bot](#-menjalankan-bot)
- [Deploy ke Heroku](#-deploy-ke-heroku)
- [Penggunaan Bot](#-penggunaan-bot)
- [Struktur Project](#-struktur-project)
- [Troubleshooting](#-troubleshooting)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## ✨ Fitur

- ✅ **Deteksi Otomatis**: Analisis gambar rongga mulut secara real-time
- 🎯 **Akurasi Tinggi**: Menggunakan MobileNetV2 pre-trained di ImageNet
- 📊 **Confidence Score**: Menampilkan tingkat kepercayaan prediksi
- 🔒 **Privacy First**: Gambar tidak disimpan di server
- 🌐 **Bahasa Indonesia**: Interface dan pesan dalam Bahasa Indonesia
- 📱 **Mobile Friendly**: Dapat diakses dari smartphone via Telegram

---

## 🔬 Cara Kerja

1. User mengirim foto rongga mulut ke bot Telegram
2. Bot memproses dan resize gambar ke 224x224 pixels
3. Model MobileNetV2 menganalisis gambar
4. Bot mengirimkan hasil (Normal/Oral Cancer) dengan confidence score
5. Bot memberikan rekomendasi berdasarkan hasil

---

## 🚀 Instalasi

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git
- Telegram Bot Token (dari [@BotFather](https://t.me/botfather))
- Dataset dari [Kaggle](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset)

### Langkah Instalasi Lokal

1. **Clone repository**

```bash
git clone <your-repo-url>
cd oral-cancer-telegram-bot
```

2. **Buat virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🎓 Training Model

### 1. Download Dataset

Download dataset dari Kaggle:
- URL: https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset
- Extract ke folder `data/`

Struktur folder yang diharapkan:
```
data/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Oral Cancer/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. Jalankan Training

```bash
python train_model.py
```

**Proses training akan:**
- Load dataset dan split train/validation (80/20)
- Melakukan data augmentation
- Training model MobileNetV2
- Optional: Fine-tuning layer terakhir
- Simpan model sebagai `oral_cancer_model.h5`
- Generate plot training history

**Parameter yang bisa diubah di `train_model.py`:**
```python
IMG_SIZE = 224          # Ukuran gambar
BATCH_SIZE = 32         # Batch size
EPOCHS = 20             # Jumlah epoch
LEARNING_RATE = 0.0001  # Learning rate
```

### 3. Hasil Training

Setelah training selesai, Anda akan mendapat:
- `oral_cancer_model.h5` - Model utama
- `oral_cancer_model.tflite` - Model versi TFLite (optional)
- `training_history.png` - Grafik training
- `best_model.h5` - Model dengan akurasi terbaik

---

## 🤖 Menjalankan Bot

### Setup Bot Token

1. **Buat bot baru di Telegram:**
   - Buka [@BotFather](https://t.me/botfather)
   - Kirim `/newbot`
   - Ikuti instruksi untuk memberi nama bot
   - Simpan token yang diberikan

2. **Set environment variable:**

```bash
# Linux/Mac
export TELEGRAM_BOT_TOKEN='your_token_here'

# Windows (Command Prompt)
set TELEGRAM_BOT_TOKEN=your_token_here

# Windows (PowerShell)
$env:TELEGRAM_BOT_TOKEN='your_token_here'
```

### Jalankan Bot Secara Lokal

```bash
python bot.py
```

Jika berhasil, Anda akan melihat:
```
🤖 Oral Cancer Detection Bot is running!
Press Ctrl+C to stop the bot
```

---

## ☁️ Deploy ke Heroku

### Prerequisites Heroku

- Akun Heroku ([daftar gratis](https://signup.heroku.com/))
- Heroku CLI ([download](https://devcenter.heroku.com/articles/heroku-cli))
- Git

### Langkah Deployment

1. **Login ke Heroku**

```bash
heroku login
```

2. **Buat aplikasi Heroku**

```bash
heroku create oral-cancer-bot-yourname
```

3. **Set environment variables**

```bash
heroku config:set TELEGRAM_BOT_TOKEN='your_token_here'
```

4. **Deploy ke Heroku**

```bash
git init
git add .
git commit -m "Initial commit"
git push heroku main
```

5. **Scale dyno**

```bash
heroku ps:scale web=1
```

6. **Check logs**

```bash
heroku logs --tail
```

### ⚠️ Catatan Penting untuk Heroku

**Batasan Heroku Free Tier:**
- Slug size max: 500MB
- RAM: 512MB
- Dyno sleep setelah 30 menit tidak aktif

**Optimisasi untuk Heroku:**

1. **Compress model** (jika terlalu besar):
```python
# Gunakan model quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

2. **Gunakan `.slugignore`** untuk exclude file besar:
```
data/
*.png
*.jpg
training_history.png
```

3. **Alternative deployment** jika model terlalu besar:
   - Google Cloud Run
   - AWS Lambda (dengan layer)
   - Railway.app
   - Render.com

---

## 📱 Penggunaan Bot

### Command Bot

- `/start` - Mulai bot dan lihat welcome message
- `/help` - Panduan lengkap penggunaan
- `/about` - Informasi tentang bot

### Cara Menggunakan

1. Buka bot di Telegram
2. Kirim command `/start`
3. Kirim foto rongga mulut yang jelas
4. Tunggu hasil analisis (3-5 detik)
5. Baca hasil dan rekomendasi

### Tips Foto Terbaik

✅ **DO:**
- Gunakan pencahayaan yang baik
- Foto close-up area mulut
- Pastikan fokus dan tidak blur
- Bersihkan area mulut sebelum foto

❌ **DON'T:**
- Foto gelap atau blur
- Jarak terlalu jauh
- Tertutup tangan/lidah
- Resolusi terlalu rendah

---

## 📁 Struktur Project

```
oral-cancer-telegram-bot/
├── bot.py                    # Main bot script
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── Procfile                  # Heroku deployment config
├── runtime.txt               # Python version for Heroku
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── data/                    # Training dataset (not in git)
│   ├── Normal/
│   └── Oral Cancer/
└── oral_cancer_model.h5     # Trained model (not in git)
```

---

## 🔧 Troubleshooting

### Model tidak ditemukan

**Error:** `Model file not found!`

**Solusi:**
```bash
# Train model terlebih dahulu
python train_model.py
```

### Bot token invalid

**Error:** `Invalid token`

**Solusi:**
- Pastikan token benar dari @BotFather
- Periksa environment variable sudah di-set
- Tidak ada spasi atau karakter tersembunyi

### Out of memory saat training

**Error:** `ResourceExhaustedError`

**Solusi:**
- Kurangi `BATCH_SIZE` di `train_model.py`
- Gunakan Google Colab untuk training (GPU gratis)
- Reduce image size

### Heroku slug too large

**Error:** `Slug size exceeds limit`

**Solusi:**
1. Buat `.slugignore`:
```
data/
*.png
*.jpg
```

2. Compress model atau gunakan TFLite

3. Remove unnecessary dependencies

### Bot tidak merespon di Heroku

**Solusi:**
```bash
# Check logs
heroku logs --tail

# Restart dyno
heroku restart

# Verify config
heroku config
```

---

## 📊 Performance Metrics

Model performance (contoh hasil training):

| Metric    | Training | Validation |
|-----------|----------|------------|
| Accuracy  | 95.2%    | 92.8%      |
| Precision | 94.8%    | 91.5%      |
| Recall    | 95.6%    | 93.2%      |
| F1-Score  | 95.2%    | 92.3%      |

*Note: Hasil aktual bervariasi tergantung dataset dan parameter training*

---

## 🔮 Future Improvements

- [ ] Multi-class classification (berbagai jenis kanker mulut)
- [ ] Webhook deployment (lebih efisien dari polling)
- [ ] Database untuk logging predictions
- [ ] Admin dashboard
- [ ] Multi-language support
- [ ] Export laporan PDF
- [ ] Integration dengan sistem rumah sakit

---

## 🤝 Contributing

Contributions are welcome! Silakan:

1. Fork repository
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## ⚠️ Disclaimer

**PENTING - BACA DENGAN TELITI:**

- Bot ini adalah **ALAT BANTU SCREENING** bukan pengganti diagnosis medis profesional
- Hasil deteksi **TIDAK BOLEH** dijadikan satu-satunya dasar untuk diagnosis
- **SELALU konsultasi** dengan dokter gigi atau spesialis untuk diagnosis dan penanganan yang tepat
- Pengembang **TIDAK BERTANGGUNG JAWAB** atas keputusan medis yang dibuat berdasarkan hasil bot
- Bot ini dikembangkan untuk **TUJUAN EDUKASI** dan penelitian
- Akurasi model dapat berbeda pada kondisi real-world
- Gambar yang dikirim ke bot diproses secara real-time dan **TIDAK DISIMPAN** di server

**Jika Anda mengalami gejala atau khawatir tentang kesehatan mulut Anda, segera hubungi tenaga medis profesional.**

---

## 👨‍💻 Developer

Dikembangkan dengan ❤️ menggunakan:
- TensorFlow/Keras
- python-telegram-bot
- MobileNetV2
- Kaggle Oral Cancer Dataset

---

## 📞 Support

Jika ada pertanyaan atau butuh bantuan:
- Buka issue di GitHub
- Email: your-email@example.com
- Telegram: @yourusername

---

## 🙏 Acknowledgments

- [Kaggle Oral Cancer Dataset](https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset)
- TensorFlow & Keras Team
- python-telegram-bot contributors
- Semua kontributor open source

---

**Selamat menggunakan! Semoga bermanfaat untuk deteksi dini kanker mulut. 🏥**

