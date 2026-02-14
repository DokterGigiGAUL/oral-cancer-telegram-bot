"""
Oral Cancer Detection Model Training Script
Uses MobileNetV2 with Transfer Learning
Dataset: https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

class OralCancerModel:
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create MobileNetV2-based transfer learning model"""
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build model
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        
        # Preprocessing
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom top layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print("Model created successfully!")
        self.model.summary()
        
        return self.model
    
    def prepare_data(self, data_dir):
        """Prepare training and validation data"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2  # 20% for validation
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        print(f"\nTraining samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def train(self, train_gen, val_gen, epochs=EPOCHS):
        """Train the model"""
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50 + "\n")
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune(self, train_gen, val_gen, epochs=10):
        """Fine-tune the model by unfreezing some layers"""
        # Unfreeze the base model
        base_model = self.model.layers[2]
        base_model.trainable = True
        
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
            loss='binary_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print("\n" + "="*50)
        print("Fine-tuning model...")
        print("="*50 + "\n")
        
        history_fine = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        return history_fine
    
    def plot_history(self, save_path='training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to {save_path}")
        
    def save_model(self, filename='oral_cancer_model.h5'):
        """Save the trained model"""
        self.model.save(filename)
        print(f"\nModel saved to {filename}")
        
        # Also save as TFLite for mobile deployment (optional)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        tflite_filename = filename.replace('.h5', '.tflite')
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_filename}")
    
    def evaluate(self, val_gen):
        """Evaluate the model"""
        print("\n" + "="*50)
        print("Evaluating model...")
        print("="*50 + "\n")
        
        results = self.model.evaluate(val_gen)
        
        print("\nFinal Results:")
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
        print(f"F1-Score: {2 * (results[2] * results[3]) / (results[2] + results[3]):.4f}")
        
        return results


def main():
    """Main training pipeline"""
    print("="*50)
    print("Oral Cancer Detection - Model Training")
    print("="*50 + "\n")
    
    # Set data directory
    # Expected structure:
    # data/
    #   ├── Normal/
    #   │   ├── image1.jpg
    #   │   └── ...
    #   └── Oral Cancer/
    #       ├── image1.jpg
    #       └── ...
    
    data_dir = 'data'  # Change this to your dataset path
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset")
        print("\nExtract it and organize as:")
        print("data/")
        print("  ├── Normal/")
        print("  └── Oral Cancer/")
        return
    
    # Initialize model
    model_trainer = OralCancerModel(img_size=IMG_SIZE)
    
    # Create model
    model_trainer.create_model()
    
    # Prepare data
    train_gen, val_gen = model_trainer.prepare_data(data_dir)
    
    # Train model
    model_trainer.train(train_gen, val_gen, epochs=EPOCHS)
    
    # Optional: Fine-tune
    response = input("\nDo you want to fine-tune the model? (y/n): ")
    if response.lower() == 'y':
        model_trainer.fine_tune(train_gen, val_gen, epochs=10)
    
    # Evaluate
    model_trainer.evaluate(val_gen)
    
    # Plot and save
    model_trainer.plot_history()
    model_trainer.save_model('oral_cancer_model.h5')
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()
