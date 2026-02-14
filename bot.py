"""
Oral Cancer Detection Telegram Bot
Detects oral cancer from images using trained AI model
"""

import os
import logging
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

# Configuration
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
MODEL_PATH = 'oral_cancer_model.h5'
IMG_SIZE = 224

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load model globally
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None


class OralCancerDetector:
    """Class to handle oral cancer detection"""
    
    def __init__(self, model, img_size=IMG_SIZE):
        self.model = model
        self.img_size = img_size
        self.class_names = ['Normal', 'Oral Cancer']
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((self.img_size, self.img_size))
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return None, None, "Model not loaded"
        
        try:
            # Preprocess
            processed_img = self.preprocess_image(image)
            
            # Predict
            prediction = self.model.predict(processed_img, verbose=0)[0][0]
            
            # Interpret results
            if prediction > 0.5:
                result = self.class_names[1]  # Oral Cancer
                confidence = prediction * 100
            else:
                result = self.class_names[0]  # Normal
                confidence = (1 - prediction) * 100
            
            return result, confidence, None
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, None, str(e)


# Initialize detector
detector = OralCancerDetector(model) if model else None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message when /start is issued"""
    welcome_message = """
🏥 *Selamat Datang di Bot Deteksi Kanker Mulut* 🏥

Bot ini menggunakan AI untuk membantu mendeteksi potensi kanker mulut dari gambar.

📸 *Cara Menggunakan:*
1. Kirim foto rongga mulut yang jelas
2. Tunggu beberapa detik untuk analisis
3. Terima hasil deteksi dan tingkat kepercayaan

⚠️ *PENTING:*
• Bot ini BUKAN pengganti diagnosis medis profesional
• Hasil hanya untuk screening awal
• Selalu konsultasi dengan dokter untuk diagnosis pasti
• Gunakan foto yang jelas dan terang

📋 *Perintah yang tersedia:*
/start - Tampilkan pesan ini
/help - Panduan penggunaan
/about - Informasi tentang bot

Kirim foto sekarang untuk memulai! 👇
    """
    
    await update.message.reply_text(
        welcome_message,
        parse_mode='Markdown'
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message when /help is issued"""
    help_message = """
📖 *Panduan Penggunaan Bot*

*1. Persiapan Foto:*
   • Gunakan pencahayaan yang baik
   • Pastikan area mulut terlihat jelas
   • Hindari foto yang blur atau gelap
   • Format: JPG, PNG, atau JPEG

*2. Mengirim Foto:*
   • Klik ikon attachment (📎)
   • Pilih foto dari galeri
   • Kirim sebagai foto (bukan file)

*3. Membaca Hasil:*
   • ✅ Normal: Tidak terdeteksi tanda-tanda kanker
   • ⚠️ Oral Cancer: Terdeteksi potensi kanker
   • Persentase menunjukkan tingkat kepercayaan AI

*4. Tips untuk Hasil Terbaik:*
   • Gunakan flash jika perlu
   • Foto dari berbagai sudut
   • Bersihkan area mulut sebelum foto
   • Ulangi jika hasil tidak yakin (<70%)

❗ *Disclaimer:*
Bot ini hanya alat bantu screening. Untuk diagnosis pasti, hubungi dokter spesialis.
    """
    
    await update.message.reply_text(
        help_message,
        parse_mode='Markdown'
    )


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send information about the bot"""
    about_message = """
ℹ️ *Tentang Bot Deteksi Kanker Mulut*

*Teknologi:*
• Model: MobileNetV2 + Transfer Learning
• Framework: TensorFlow/Keras
• Platform: Python + python-telegram-bot

*Dataset:*
Dataset yang digunakan untuk training berasal dari Kaggle - Oral Cancer Dataset yang berisi ribuan gambar rongga mulut normal dan terdampak kanker.

*Akurasi:*
Model telah dilatih dengan data augmentation dan mencapai akurasi validasi tinggi. Namun, akurasi tidak 100% dan bisa berbeda pada kondisi real-world.

*Pengembang:*
Bot ini dikembangkan untuk tujuan edukasi dan screening awal. Tidak boleh digunakan sebagai satu-satunya dasar diagnosis.

*Privasi:*
• Foto tidak disimpan di server
• Semua proses dilakukan real-time
• Data tidak dibagikan ke pihak ketiga

*Sumber Kode:*
Open source - dapat dikembangkan lebih lanjut

📧 Untuk pertanyaan atau feedback, hubungi pengembang.
    """
    
    await update.message.reply_text(
        about_message,
        parse_mode='Markdown'
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages from users"""
    
    if detector is None or detector.model is None:
        await update.message.reply_text(
            "❌ Maaf, model AI sedang tidak tersedia. Silakan coba lagi nanti."
        )
        return
    
    # Send processing message
    processing_msg = await update.message.reply_text(
        "🔍 Menganalisis gambar...\nMohon tunggu sebentar..."
    )
    
    try:
        # Get the photo
        photo = await update.message.photo[-1].get_file()
        
        # Download photo as bytes
        photo_bytes = await photo.download_as_bytearray()
        
        # Convert to PIL Image
        image = Image.open(BytesIO(photo_bytes))
        
        # Make prediction
        result, confidence, error = detector.predict(image)
        
        if error:
            await processing_msg.edit_text(
                f"❌ Terjadi kesalahan saat analisis:\n{error}"
            )
            return
        
        # Prepare result message
        if result == 'Normal':
            emoji = "✅"
            status = "NORMAL"
            recommendation = (
                "\n\n💡 *Rekomendasi:*\n"
                "Kondisi terlihat normal. Tetap jaga kesehatan mulut dengan:\n"
                "• Sikat gigi 2x sehari\n"
                "• Hindari rokok dan alkohol\n"
                "• Periksa gigi rutin setiap 6 bulan"
            )
        else:
            emoji = "⚠️"
            status = "POTENSI KANKER MULUT"
            recommendation = (
                "\n\n🏥 *Rekomendasi PENTING:*\n"
                "Terdeteksi potensi kanker mulut. SEGERA:\n"
                "1. Konsultasi dengan dokter gigi/spesialis\n"
                "2. Lakukan pemeriksaan lebih lanjut\n"
                "3. Jangan panik, deteksi dini tingkatkan kesembuhan\n"
                "4. Bawa hasil ini saat konsultasi"
            )
        
        # Confidence level interpretation
        if confidence >= 90:
            confidence_level = "Sangat Tinggi"
        elif confidence >= 75:
            confidence_level = "Tinggi"
        elif confidence >= 60:
            confidence_level = "Sedang"
        else:
            confidence_level = "Rendah"
        
        result_message = f"""
{emoji} *HASIL DETEKSI* {emoji}

*Status:* {status}
*Kepercayaan:* {confidence:.2f}% ({confidence_level})

📊 *Interpretasi:*
Model AI mendeteksi bahwa gambar termasuk kategori "{result}" dengan tingkat kepercayaan {confidence:.1f}%.
{recommendation}

⚠️ *DISCLAIMER:*
Hasil ini BUKAN diagnosis medis resmi. Selalu konsultasi dengan tenaga medis profesional untuk diagnosis dan penanganan yang tepat.

Kirim foto lain untuk analisis tambahan.
        """
        
        await processing_msg.edit_text(
            result_message,
            parse_mode='Markdown'
        )
        
        logger.info(f"Prediction made: {result} ({confidence:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await processing_msg.edit_text(
            f"❌ Terjadi kesalahan saat memproses foto:\n{str(e)}\n\n"
            "Silakan coba kirim foto lain atau hubungi pengembang."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    await update.message.reply_text(
        "📸 Silakan kirim *foto* rongga mulut untuk dianalisis.\n\n"
        "Gunakan /help untuk panduan lengkap.",
        parse_mode='Markdown'
    )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")
    
    if update and update.message:
        await update.message.reply_text(
            "❌ Terjadi kesalahan internal. Silakan coba lagi nanti."
        )


def main():
    """Start the bot"""
    
    if BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        logger.error("Please set TELEGRAM_BOT_TOKEN environment variable!")
        print("\n❌ ERROR: Bot token not configured!")
        print("\nPlease set your Telegram bot token:")
        print("  export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("\nOr edit the BOT_TOKEN variable in the script.")
        return
    
    if model is None:
        logger.error("Model not loaded! Please train the model first.")
        print("\n❌ ERROR: Model file not found!")
        print(f"\nPlease ensure '{MODEL_PATH}' exists.")
        print("Run train_model.py first to train the model.")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start bot
    logger.info("Bot started successfully!")
    print("\n" + "="*50)
    print("🤖 Oral Cancer Detection Bot is running!")
    print("="*50)
    print("\nPress Ctrl+C to stop the bot\n")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
