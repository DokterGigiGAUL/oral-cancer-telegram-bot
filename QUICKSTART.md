# 🚀 Panduan Cepat - Oral Cancer Detection Bot

Panduan singkat untuk memulai bot deteksi kanker mulut.

---

## 📝 Checklist Persiapan

- [ ] Python 3.11+ terinstall
- [ ] pip terinstall
- [ ] Bot token dari @BotFather
- [ ] Dataset dari Kaggle sudah didownload
- [ ] Akun Heroku (untuk deployment)

---

## ⚡ Quick Start (5 Langkah)

### 1️⃣ Clone & Install

```bash
git clone <your-repo>
cd oral-cancer-telegram-bot
pip install -r requirements.txt
```

### 2️⃣ Setup Dataset

```bash
# Download dari: https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset
# Extract ke folder data/

# Struktur harus seperti ini:
data/
├── Normal/
└── Oral Cancer/
```

### 3️⃣ Training Model

```bash
python train_model.py
```

**Waktu:** ~20-30 menit (tergantung hardware)
**Output:** `oral_cancer_model.h5`

### 4️⃣ Setup Bot Token

```bash
# Dapatkan token dari @BotFather di Telegram
# Lalu set environment variable:

# Linux/Mac:
export TELEGRAM_BOT_TOKEN='your_token_here'

# Windows:
set TELEGRAM_BOT_TOKEN=your_token_here
```

### 5️⃣ Jalankan Bot

```bash
python bot.py
```

**Selesai!** Bot sudah berjalan. Buka Telegram dan coba kirim foto!

---

## ☁️ Deploy ke Heroku (3 Langkah)

### 1️⃣ Setup Heroku

```bash
heroku login
heroku create nama-bot-anda
```

### 2️⃣ Set Token

```bash
heroku config:set TELEGRAM_BOT_TOKEN='your_token_here'
```

### 3️⃣ Deploy

```bash
git init
git add .
git commit -m "Initial commit"
git push heroku main
heroku ps:scale web=1
```

**Selesai!** Bot online 24/7 di Heroku!

---

## 🧪 Testing Bot

1. Buka bot di Telegram
2. Kirim `/start`
3. Kirim foto rongga mulut
4. Lihat hasil analisis

---

## 🆘 Troubleshooting Cepat

| Masalah | Solusi |
|---------|--------|
| Model not found | Jalankan `python train_model.py` dulu |
| Invalid token | Cek token dari @BotFather, pastikan benar |
| Bot tidak respon | Cek `heroku logs --tail` |
| Out of memory | Kurangi BATCH_SIZE di train_model.py |

---

## 📚 Dokumentasi Lengkap

Lihat **README.md** untuk dokumentasi lengkap dan detail.

---

## 💡 Tips

- **Training pertama kali?** Gunakan Google Colab (gratis GPU)
- **Model terlalu besar?** Compress dengan TFLite
- **Heroku error?** Coba Railway.app atau Render.com sebagai alternatif
- **Ingin akurasi lebih tinggi?** Tambah epoch atau gunakan EfficientNet

---

## 🎯 Next Steps

Setelah bot berjalan:

1. ✅ Test dengan berbagai foto
2. ✅ Monitor akurasi predictions
3. ✅ Improve model jika perlu
4. ✅ Tambah fitur (database, admin panel, dll)
5. ✅ Share ke komunitas!

---

**Selamat coding! 🚀**

Jika ada pertanyaan, buka issue di GitHub atau hubungi developer.
