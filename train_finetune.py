import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- Configuration ---
BASE_MODEL_PATH = 'vgg_unfrozen.h5'      # Pretrained base model
FINETUNED_MODEL_PATH = 'vgg_finetuned.h5'  # Path to save fine-tuned model
IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32
FINETUNE_EPOCHS = 15
INITIAL_EPOCHS = 18  # The number of epochs the base model was trained for

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

# --- Compute Class Weights to handle imbalance ---
classes = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# --- Fine-Tuning Logic ---
def fine_tune_model():
    # Load the previously trained model
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"Error: Base model not found at {BASE_MODEL_PATH}")
        return

    print(f"Loading base model from: {BASE_MODEL_PATH}")
    model = load_model(BASE_MODEL_PATH)

    # Freeze all layers first
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze last VGG block (block5) for fine-tuning
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True

    # Optional: Freeze BatchNorm layers to prevent instability
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    print("\nTrainable layers after unfreezing block5:")
    for layer in model.layers:
        print(f"{layer.name}: {layer.trainable}")

    # Compile model with very low learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(FINETUNED_MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Fine-tune the model
    history = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS + FINETUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        validation_data=validation_generator,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    print("\nFine-tuning complete.")
    print(f"Fine-tuned model saved to: {FINETUNED_MODEL_PATH}")

if __name__ == '__main__':
    fine_tune_model()
