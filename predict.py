import mlflow
import pandas as pd
import numpy as np
import argparse

"""
Test prediction using the MLflow package model
"""

def test_local_env(input: np.ndarray):
    """
    Test prediction using the MLflow package model running in the local (active) virtual environment.
    """
    
    model = mlflow.pyfunc.load_model("mlflow_model")
    predictions = model.predict(input)
    print(predictions)

def test_isolated_env(input: np.ndarray):
    """
    Test prediction using the MLflow package model in an isolated virtual environment.
    """
    
    predictions = mlflow.models.predict(
        model_uri="mlflow_model",
        input_data=input,
        env_manager="uv",
    )
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow model predictions.")
    parser.add_argument(
        "--local", action="store_true",
        help="Run prediction in the local environment (default: False)."
    )
    parser.add_argument(
        "--isolated", action="store_true",
        help="Run prediction in an isolated environment (default: False)."
    )
    args = parser.parse_args()

    if not args.local and not args.isolated:
        args.local = True
        
    input = np.random.rand(256, 256, 256).astype(np.float32)

    if args.local:
        print("Running prediction in the local Python environment.")
        test_local_env(input)
    if args.isolated:
        print("Running prediction in an isolated Python environment.")
        test_isolated_env(input)