"""Lightweight stub for tensorflow_decision_forests.

The tensorflowjs converter imports this package unconditionally in a SavedModel
conversion path, but our project only converts a plain Keras `.h5` file.
Providing this stub avoids pulling a very large optional dependency.
"""
