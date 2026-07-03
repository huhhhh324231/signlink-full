# ============ PROJECT CONFIG ============

# ============ MODEL ============
MODEL_CONFIG = {
    "base_model": "MobileNetV2",
    "input_size": (224, 224),
    "freeze_base": True,
    "dense_layers": [256, 128],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 10,
    "validation_split": 0.2,
    "early_stopping_patience": 3,
}

# ============ CAMERA ============
CAMERA_CONFIG = {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "flip_horizontal": True,
}

# ============ INFERENCE ============
INFERENCE_CONFIG = {
    "confidence_threshold": 0.65,
    "buffer_size": 7,
    "min_consensus_frames": 4,
    "hand_padding": 30,
    "show_pose": True,
    "show_fps": True,
    "show_confidence": True,
}

# ============ MEDIAPIPE HANDS ============
HANDS_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.6,
}

# ============ SIGN LABELS ============
SIGN_LABELS = {i: chr(ord("a") + i) for i in range(26)}

# ============ DATA ============
DATA_CONFIG = {
    "data_dir": "sign_language_data",
    "model_save_path": "sign_language_model.h5",
    "samples_per_sign": 100,
    "img_size": (224, 224),
    "train_test_split": 0.8,
}

# ============ TRAINING ============
ADVANCED_CONFIG = {
    "use_data_augmentation": True,
    "augmentation": {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "horizontal_flip": True,
        "zoom_range": 0.2,
    },
    "use_tensorboard": False,
    "checkpoint_dir": "./checkpoints",
}


def get_model_config():
    return MODEL_CONFIG


def get_camera_config():
    return CAMERA_CONFIG


def get_inference_config():
    return INFERENCE_CONFIG


def get_hands_config():
    return HANDS_CONFIG


def get_sign_labels():
    return SIGN_LABELS


def get_data_config():
    return DATA_CONFIG


def get_advanced_config():
    return ADVANCED_CONFIG


if __name__ == "__main__":
    print("PROJECT CONFIG")
    print("=" * 60)
    print("\nModel:", MODEL_CONFIG)
    print("\nCamera:", CAMERA_CONFIG)
    print("\nInference:", INFERENCE_CONFIG)
    print("\nLabels:", SIGN_LABELS)
