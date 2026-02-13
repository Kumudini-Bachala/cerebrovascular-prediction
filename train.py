import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Constants
MODEL_FILENAME = 'vgg_unfrozen.h5'  # Consistent model filename

def check_dataset():
    yes_path = 'img/Stroke'
    no_path = 'img/Normal'
    
    yes_files = os.listdir(yes_path)
    no_files = os.listdir(no_path)
    
    print(f"Number of Stroke images: {len(yes_files)}")
    print(f"Number of Normal images: {len(no_files)}")
    return len(yes_files), len(no_files)

def create_model():
    # Create the model
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(240, 240, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4608, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1152, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    print("TensorFlow version:", tf.__version__)
    
    # Check dataset
    print("\nChecking dataset...")
    n_stroke, n_normal = check_dataset()
    total = n_stroke + n_normal

    # Calculate class weights
    # The classes are ordered alphabetically by the generator: Normal (0), Stroke (1)
    weight_for_0 = (1 / n_normal) * (total / 2.0)
    weight_for_1 = (1 / n_stroke) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"\nClass weights calculated:")
    print(f"  Weight for Normal (class 0): {class_weight[0]:.2f}")
    print(f"  Weight for Stroke (class 1): {class_weight[1]:.2f}")

    # Set up data generators
    print("\nSetting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )

    # Create training and validation generators
    train_generator = train_datagen.flow_from_directory(
        'img',
        target_size=(240, 240),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        'img',
        target_size=(240, 240),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    print("\nCreating and compiling model...")
    model = create_model()
    model.summary()

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(MODEL_FILENAME, monitor='val_loss', save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Train the model
    print("\nStarting training...")
    
    # Custom callback to log detailed progress
    class DetailedProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            with open('training_progress.txt', 'a') as f:
                f.write(f"\nStarting Epoch {epoch + 1}\n")
                f.flush()
        
        def on_batch_end(self, batch, logs=None):
            if batch % 10 == 0:  # Log every 10 batches
                with open('training_progress.txt', 'a') as f:
                    f.write(f"Batch {batch}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}\n")
                    f.flush()
        
        def on_epoch_end(self, epoch, logs=None):
            with open('training_progress.txt', 'a') as f:
                f.write(f"\nEpoch {epoch + 1} Results:\n")
                f.write(f"Training Accuracy: {logs['accuracy']:.4f}\n")
                f.write(f"Training Loss: {logs['loss']:.4f}\n")
                f.write(f"Validation Accuracy: {logs['val_accuracy']:.4f}\n")
                f.write(f"Validation Loss: {logs['val_loss']:.4f}\n")
                f.write("-" * 50 + "\n")
                f.flush()
            
            # Save intermediate model
            if (epoch + 1) % 2 == 0:  # Save every 2 epochs
                self.model.save(f'model_epoch_{epoch+1}.h5')
                with open('training_progress.txt', 'a') as f:
                    f.write(f"Saved intermediate model at epoch {epoch+1}\n")
                    f.flush()
    
    # Clear previous progress file
    with open('training_progress.txt', 'w') as f:
        f.write("Starting Training\n")
        f.write("=" * 50 + "\n")
    
    callbacks.append(DetailedProgressCallback())
    
    history = model.fit(
        train_generator,
        epochs=25,
        validation_data=validation_generator,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # Plot and save training history
    print("\nSaving training history plot...")
    plot_training_history(history)

    # Save the final model
    print("\nSaving model...")
    model.save(MODEL_FILENAME)
    
    # Also save the best weights from training
    if os.path.exists('model.h5'):
        os.replace('model.h5', MODEL_FILENAME)
    
    print("Model saved successfully!")
    print(f"Model saved as: {MODEL_FILENAME}")


if __name__ == "__main__":
    main()