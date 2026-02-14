"""
Test script for Oral Cancer Detection Model
Use this to test your trained model with sample images
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

# Configuration
MODEL_PATH = 'oral_cancer_model.h5'
IMG_SIZE = 224

def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        print("\nPlease train the model first using:")
        print("  python train_model.py")
        return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array, None
    
    except Exception as e:
        return None, str(e)

def predict_image(model, image_path):
    """Make prediction on a single image"""
    print(f"\n📸 Testing image: {image_path}")
    
    # Preprocess
    img_array, error = preprocess_image(image_path)
    if error:
        print(f"❌ Error preprocessing image: {error}")
        return
    
    # Predict
    try:
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Interpret
        class_names = ['Normal', 'Oral Cancer']
        
        if prediction > 0.5:
            result = class_names[1]
            confidence = prediction * 100
        else:
            result = class_names[0]
            confidence = (1 - prediction) * 100
        
        # Display results
        print("="*50)
        print(f"🎯 Result: {result}")
        print(f"📊 Confidence: {confidence:.2f}%")
        print(f"📈 Raw prediction: {prediction:.4f}")
        print("="*50)
        
        # Interpretation
        if result == 'Normal':
            print("✅ No signs of oral cancer detected")
        else:
            print("⚠️  Potential oral cancer detected")
            print("⚠️  Please consult with a medical professional")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

def test_model_summary(model):
    """Display model summary and information"""
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    
    print(f"\nModel type: Transfer Learning (MobileNetV2)")
    print(f"Input size: {IMG_SIZE}x{IMG_SIZE}x3")
    print(f"Output: Binary classification (Normal/Oral Cancer)")
    
    print("\n" + "-"*50)
    print("MODEL ARCHITECTURE")
    print("-"*50)
    model.summary()
    
    print("\n" + "-"*50)
    print("TOTAL PARAMETERS")
    print("-"*50)
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024*1024):.2f} MB")

def interactive_test(model):
    """Interactive testing mode"""
    print("\n" + "="*50)
    print("INTERACTIVE TESTING MODE")
    print("="*50)
    print("\nEnter image path to test (or 'q' to quit)")
    
    while True:
        print("\n" + "-"*50)
        image_path = input("Image path: ").strip()
        
        if image_path.lower() in ['q', 'quit', 'exit']:
            print("\n👋 Exiting...")
            break
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            continue
        
        predict_image(model, image_path)

def batch_test(model, folder_path):
    """Test all images in a folder"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\n📁 Testing {len(image_files)} images from: {folder_path}")
    
    results = []
    for img_path in image_files:
        predict_image(model, img_path)
        
        # Simple aggregation (you can enhance this)
        print()  # spacing

def main():
    """Main function"""
    print("="*50)
    print("🧪 ORAL CANCER MODEL TESTER")
    print("="*50)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Show model info
    test_model_summary(model)
    
    # Check arguments
    if len(sys.argv) > 1:
        # Test mode from command line
        test_path = sys.argv[1]
        
        if os.path.isfile(test_path):
            # Single image
            predict_image(model, test_path)
        elif os.path.isdir(test_path):
            # Batch test folder
            batch_test(model, test_path)
        else:
            print(f"❌ Invalid path: {test_path}")
    else:
        # Interactive mode
        interactive_test(model)

if __name__ == "__main__":
    print("\n💡 Usage:")
    print("  python test_model.py                    # Interactive mode")
    print("  python test_model.py image.jpg          # Test single image")
    print("  python test_model.py test_folder/       # Test all images in folder")
    print()
    
    main()
