# -*- coding: utf-8 -*-
"""
Train hand gesture recognition model using local sign_language_data.
This version keeps CPU-friendly defaults but adds a light fine-tuning stage.
"""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("[*] Loading configuration...")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "sign_language_data"
MODEL_OUTPUT = BASE_DIR / "sign_language_model.h5"
IMG_SIZE = (
    int(os.getenv("IMG_SIZE", "224")),
    int(os.getenv("IMG_SIZE", "224")),
)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
EPOCHS_HEAD = int(os.getenv("EPOCHS_HEAD", "8"))
EPOCHS_FINE_TUNE = int(os.getenv("EPOCHS_FINE_TUNE", "4"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.2"))
FINE_TUNE_AT = int(os.getenv("FINE_TUNE_AT", "120"))

print(
    f"[*] Settings: img_size={IMG_SIZE}, batch_size={BATCH_SIZE}, "
    f"epochs_head={EPOCHS_HEAD}, epochs_fine_tune={EPOCHS_FINE_TUNE}, "
    f"validation_split={VALIDATION_SPLIT}, fine_tune_at={FINE_TUNE_AT}"
)

gesture_classes = sorted(
    [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(DATA_DIR / d) and len(d) == 1 and d.isalpha()
    ]
)
gesture_classes = gesture_classes[:26]
num_classes = len(gesture_classes)

print(f"[*] Found {num_classes} gesture classes: {gesture_classes}")

if num_classes == 0:
    raise ValueError("[!] No valid gesture folders found in sign_language_data/")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=18,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.15,
    shear_range=0.1,
    brightness_range=(0.8, 1.2),
    validation_split=VALIDATION_SPLIT,
)

eval_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT,
)

print("[*] Setting up data generators...")
train_generator = train_datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=gesture_classes,
    subset="training",
)

val_generator = eval_datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=gesture_classes,
    subset="validation",
)

print(f"[*] Training samples: {train_generator.samples}")
print(f"[*] Validation samples: {val_generator.samples}")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ]
)


def build_callbacks(stage_name):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUTPUT),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(BASE_DIR / f"training_{stage_name}.log"), append=False),
    ]


def compile_model(learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


train_steps = max(1, train_generator.samples // BATCH_SIZE)
val_steps = max(1, val_generator.samples // BATCH_SIZE)

print("[*] Stage 1: training classifier head...")
compile_model(0.001)
history_head = model.fit(
    train_generator,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=build_callbacks("head"),
    verbose=1,
)

if EPOCHS_FINE_TUNE > 0:
    print("[*] Stage 2: fine-tuning top MobileNetV2 layers...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    compile_model(0.0001)
    history_fine = model.fit(
        train_generator,
        epochs=EPOCHS_FINE_TUNE,
        steps_per_epoch=train_steps,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=build_callbacks("fine_tune"),
        verbose=1,
    )
    final_history = history_fine
else:
    final_history = history_head

print(f"[*] Saving final model to {MODEL_OUTPUT}...")
model.save(str(MODEL_OUTPUT))
print("[*] Training completed!")

print(f"\n[*] Final Training Accuracy: {final_history.history['accuracy'][-1]:.4f}")
print(f"[*] Final Validation Accuracy: {final_history.history['val_accuracy'][-1]:.4f}")
