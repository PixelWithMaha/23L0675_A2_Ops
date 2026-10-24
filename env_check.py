import sys
import tensorflow as tf
import mlflow
import dagshub
import dvc
import PIL
import pandas as pd
import numpy as np

def run_env_check():
    print("=" * 60)
    print("VISIONGUARD ENVIRONMENT & HARDWARE AUDIT")
    print("=" * 60)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"TensorFlow        : {tf.__version__}")
    print(f"MLflow            : {mlflow.__version__}")
    print(f"DagsHub           : {dagshub.__version__}")
    print(f"Pillow (PIL)      : {PIL.__version__}")
    print(f"Pandas            : {pd.__version__}")
    print(f"NumPy             : {np.__version__}")
    print("-" * 60)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU Acceleration  : ENABLED ({len(gpus)} GPU(s) detected)")
        for gpu in gpus:
            print(f"  └── Device Name : {gpu.name}")
    else:
        print("GPU Acceleration  : NOT DETECTED (Running on CPU)")
    print("=" * 60)

if __name__ == "__main__":
    run_env_check()