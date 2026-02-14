# 🚂 Alternative Deployment Options

Jika Heroku tidak cocok atau mengalami masalah, berikut adalah alternatif deployment gratis lainnya.

---

## 🚂 Railway.app

Railway adalah platform deployment modern dengan free tier yang bagus.

### Kelebihan Railway:
- ✅ Free tier lebih generous (500 jam/bulan)
- ✅ Tidak sleep otomatis seperti Heroku
- ✅ Setup lebih mudah
- ✅ Database gratis included
- ✅ Support environment variables

### Cara Deploy ke Railway:

1. **Daftar di Railway**
   - Kunjungi: https://railway.app
   - Sign up dengan GitHub

2. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   ```

3. **Login**
   ```bash
   railway login
   ```

4. **Initialize Project**
   ```bash
   railway init
   ```

5. **Add Environment Variable**
   ```bash
   railway variables set TELEGRAM_BOT_TOKEN='your_token_here'
   ```

6. **Deploy**
   ```bash
   railway up
   ```

7. **Monitor**
   ```bash
   railway logs
   ```

### Railway Configuration

Buat file `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python bot.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

---

## 🎨 Render.com

Render adalah alternatif bagus lainnya dengan UI yang user-friendly.

### Kelebihan Render:
- ✅ Free tier tersedia
- ✅ Auto-deploy dari GitHub
- ✅ Built-in logging
- ✅ Custom domains gratis
- ✅ Zero config deployment

### Cara Deploy ke Render:

1. **Daftar di Render**
   - Kunjungi: https://render.com
   - Sign up dengan GitHub

2. **Create New Web Service**
   - Klik "New +"
   - Pilih "Web Service"
   - Connect repository Anda

3. **Configure Service**
   - **Name:** oral-cancer-bot
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python bot.py`
   - **Plan:** Free

4. **Add Environment Variable**
   - Scroll ke "Environment Variables"
   - Add: `TELEGRAM_BOT_TOKEN` = your_token

5. **Deploy**
   - Klik "Create Web Service"
   - Tunggu build selesai

### Render Configuration

Buat file `render.yaml`:
```yaml
services:
  - type: web
    name: oral-cancer-bot
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python bot.py
    plan: free
    envVars:
      - key: TELEGRAM_BOT_TOKEN
        sync: false
```

---

## 🐳 Google Cloud Run

Untuk yang butuh scalability lebih tinggi.

### Kelebihan Google Cloud Run:
- ✅ Pay per use (gratis untuk low traffic)
- ✅ Auto-scaling
- ✅ Container-based
- ✅ Integration dengan GCP services

### Cara Deploy ke Cloud Run:

1. **Install Google Cloud SDK**
   ```bash
   # Download dari: https://cloud.google.com/sdk/docs/install
   ```

2. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   CMD ["python", "bot.py"]
   ```

3. **Build & Deploy**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/oral-cancer-bot
   gcloud run deploy --image gcr.io/PROJECT-ID/oral-cancer-bot --platform managed
   ```

4. **Set Environment Variable**
   ```bash
   gcloud run services update oral-cancer-bot \
     --set-env-vars TELEGRAM_BOT_TOKEN=your_token
   ```

---

## 🔧 Fly.io

Platform modern untuk deployment aplikasi.

### Kelebihan Fly.io:
- ✅ Free tier available
- ✅ Global deployment
- ✅ Low latency
- ✅ Dockerfile support

### Cara Deploy ke Fly.io:

1. **Install flyctl**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**
   ```bash
   flyctl auth login
   ```

3. **Launch App**
   ```bash
   flyctl launch
   ```

4. **Set Secret**
   ```bash
   flyctl secrets set TELEGRAM_BOT_TOKEN=your_token
   ```

5. **Deploy**
   ```bash
   flyctl deploy
   ```

---

## 📊 Perbandingan Platform

| Platform | Free Tier | Sleep | Build Time | Complexity |
|----------|-----------|-------|------------|------------|
| Heroku | 550 hrs/mo | Ya (30 min) | Medium | Easy |
| Railway | 500 hrs/mo | Tidak | Fast | Very Easy |
| Render | Unlimited | Ya | Medium | Easy |
| Cloud Run | Pay per use | Ya | Fast | Medium |
| Fly.io | Limited | Tidak | Fast | Medium |

---

## 💡 Rekomendasi

**Untuk Pemula:** Railway atau Render
- Setup paling mudah
- UI user-friendly
- Documentation bagus

**Untuk Production:** Google Cloud Run atau Fly.io
- Lebih reliable
- Better performance
- Scalable

**Untuk Belajar:** Heroku
- Paling populer
- Banyak tutorial
- Community besar

---

## 🆘 Troubleshooting

### Railway Issues

**Problem:** Build failed
```bash
railway logs --tail
```

**Solution:** Check Python version di railway.json

### Render Issues

**Problem:** Service tidak start
- Check start command: `python bot.py`
- Verify environment variables
- Check logs di dashboard

### Cloud Run Issues

**Problem:** Container tidak running
```bash
gcloud run services logs read oral-cancer-bot --limit=50
```

**Solution:** Pastikan Dockerfile correct dan port exposed

---

## 📚 Resources

- **Railway:** https://docs.railway.app
- **Render:** https://render.com/docs
- **Cloud Run:** https://cloud.google.com/run/docs
- **Fly.io:** https://fly.io/docs

---

## ✅ Checklist Deployment

Apapun platform yang dipilih:

- [ ] Model file ada dan ukuran < 500MB
- [ ] requirements.txt lengkap
- [ ] Bot token sudah di-set
- [ ] Test lokal dulu sebelum deploy
- [ ] Monitor logs setelah deploy
- [ ] Test bot di Telegram
- [ ] Setup monitoring/alerts

---

**Good luck dengan deployment! 🚀**

Jika stuck, cek logs dan baca dokumentasi platform yang dipilih.
