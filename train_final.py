import os
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
FINAL_MODEL_PATH = 'brain_stroke_final_model.h5'
IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINETUNE_EPOCHS = 30

# --- Data Generators ---
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

train_generator = train_datagen.flow_from_directory(
    'img',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    'img',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- Model Architecture ---
def build_model():
    """
    Builds a new, more efficient model using VGG19 and GlobalAveragePooling2D.
    """
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Freeze the base model initially
    base_model.trainable = False

    # Add a more efficient head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x) # Add dropout for regularization
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# --- Main Training Logic ---
def main():
    # Build the model
    model = build_model()

    # STAGE 1: Feature Extraction
    print("--- STAGE 1: Training the top layers ---")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    history = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # STAGE 2: Fine-Tuning
    print("\n--- STAGE 2: Fine-tuning the top blocks of VGG19 ---")
    
    # Unfreeze the top two blocks (block4 and block5)
    base_model = model.layers[0]
    base_model.trainable = True
    
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block4_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # Re-compile with a very low learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add ModelCheckpoint to save the absolute best model
    fine_tune_callbacks = [
        ModelCheckpoint(FINAL_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True), # More patience for fine-tuning
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]

    # Continue training
    total_epochs = INITIAL_EPOCHS + FINETUNE_EPOCHS
    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        callbacks=fine_tune_callbacks
    )

    print("\n--- Final Training Complete ---")
    print(f"The best model has been saved to: {FINAL_MODEL_PATH}")

if __name__ == '__main__':
    main()
