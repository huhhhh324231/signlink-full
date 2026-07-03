"""Export a Keras .h5 model to a minimal TensorFlow.js layers-model format.

This avoids the heavyweight optional dependencies pulled in by the latest
tensorflowjs Python package while still producing a standard `model.json`
plus binary weight shard for browser inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

BASE_DIR = Path(__file__).resolve().parent
INPUT_MODEL = BASE_DIR / "sign_language_model.h5"
OUTPUT_DIR = BASE_DIR / "signlink-ai" / "public" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS = [chr(ord("A") + i) for i in range(26)]


def normalize_dtype(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float64:
        return array.astype(np.float32)
    if array.dtype == np.int64:
        return array.astype(np.int32)
    return array


def tfjs_dtype(array: np.ndarray) -> str:
    kind = array.dtype.kind
    if kind == "f":
        return "float32" if array.dtype != np.float16 else "float16"
    if kind == "i":
        return "int32"
    if kind == "u":
        return "uint8" if array.dtype.itemsize == 1 else "uint16"
    if kind == "b":
        return "bool"
    raise ValueError(f"Unsupported dtype for TF.js export: {array.dtype}")


def main() -> None:
    print(f"[*] Loading model: {INPUT_MODEL}")
    model = tf.keras.models.load_model(INPUT_MODEL)

    topology = json.loads(model.to_json())
    weights = []
    blob_parts = []

    for variable, value in zip(model.weights, model.get_weights()):
        array = normalize_dtype(np.asarray(value))
        weights.append(
            {
                "name": variable.name.replace(":0", ""),
                "shape": list(array.shape),
                "dtype": tfjs_dtype(array),
            }
        )
        blob_parts.append(array.tobytes(order="C"))

    weights_path = OUTPUT_DIR / "group1-shard1of1.bin"
    weights_path.write_bytes(b"".join(blob_parts))

    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{getattr(keras, '__version__', '2.x')}",
        "convertedBy": "custom-export-script",
        "modelTopology": topology,
        "weightsManifest": [
            {
                "paths": [weights_path.name],
                "weights": weights,
            }
        ],
    }

    model_json_path = OUTPUT_DIR / "model.json"
    model_json_path.write_text(json.dumps(model_json), encoding="utf-8")

    metadata = {
        "labels": LABELS,
        "inputSize": [224, 224],
        "normalization": "rgb/255.0",
        "sourceModel": INPUT_MODEL.name,
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[*] Exported TF.js model to: {OUTPUT_DIR}")
    print(f"[*] Weights: {weights_path.name}")
    print(f"[*] Model JSON: {model_json_path.name}")
    print(f"[*] Metadata: {metadata_path.name}")
    print(f"[*] Input shape: {model.input_shape}")
    print(f"[*] Output shape: {model.output_shape}")


if __name__ == "__main__":
    main()
