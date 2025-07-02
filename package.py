import shutil
import numpy as np
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec, ColSpec

from model_code.mlflow_model import MyMLflowModel

def main():
    # Create example input matching the model's requirements (3D numpy array)
    input_example = np.random.rand(256, 256, 256).astype(np.float32)  # Z,Y,X format
    
    # Define model signature with proper numpy dtype objects
    input_schema = Schema([
        TensorSpec(type=np.dtype('float32'), shape=(-1, -1, -1))  # Variable size 3D volume
    ])
    output_schema = Schema([ColSpec("double") for i in range(3)] + [ColSpec("string")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Model artifacts
    artifacts = {
        "pytorch_model": "model_data/model_weights.pt",
        "labels_file": "model_data/labels.json"
    }

    # Clean previous model if exists
    mlflow_model_path = "mlflow_model"
    shutil.rmtree(mlflow_model_path, ignore_errors=True)
    
    # Package the model
    mlflow.pyfunc.save_model(
        path=mlflow_model_path,
        python_model=MyMLflowModel(),
        artifacts=artifacts,
        code_paths=["model_code"],
        input_example=input_example,
        signature=signature,
        pip_requirements="requirements.txt",
        metadata={
            "model_type": "3D_particle_detection",
            "input_description": "3D numpy array (Z,Y,X) of electron density values",
            "output_description": "Detected particle coordinates with classifications"
        }
    )

    print(f"MLflow model packaged at: {mlflow_model_path}")

if __name__ == "__main__":
    main()