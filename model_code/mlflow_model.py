import mlflow
import torch
import numpy as np
import json
import cc3d
import pandas as pd
from torch.utils.data import DataLoader
from pandas import DataFrame

from model_code.pytorch_model import EnsembleTTABPD
from model_code.patch_dataset import PatchDataset

class MyMLflowModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model for 3D particle detection and classification.
    Processes input volumes through:
    1. Volume normalization
    2. Patch-based inference
    3. Probability map reconstruction
    4. Connected components analysis
    5. Particle coordinate extraction
    """

    def load_context(self, context):
        """
        Initialize model and load artifacts when loaded by MLflow.
        
        Args:
            context: MLflow context containing:
                - pytorch_model: Model weights file
                - labels_file: JSON config with particle detection thresholds
        """
        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize and load the ensemble model
        self.model = EnsembleTTABPD().to(self.device)
        self.model.load_state_dict(
            torch.load(context.artifacts["pytorch_model"], 
                     map_location=self.device)
        )
        self.model.eval()  # Set to evaluation mode

        # Load particle detection parameters
        with open(context.artifacts["labels_file"], "r") as f:
            self.labels = json.load(f)

    def _preprocess(self, model_input: np.ndarray) -> DataLoader:
        """
        Normalize volume and prepare patch-based DataLoader.
        
        Args:
            model_input: 3D numpy array (Z,Y,X) of input volume
            
        Returns:
            DataLoader yielding 128x128x128 patches with coordinates
        """
        # Normalize to [0,1] range using precomputed min/max
        pmin, pmax = -1.1769942479337e-05, 1.2801160441345688e-05
        model_input = (model_input - pmin) / (pmax - pmin)
        
        return DataLoader(
            PatchDataset(model_input, [96, 96, 96]),
            batch_size=1,
            shuffle=False,
            num_workers=2
        )

    def _get_probabilities(self, data_loader: DataLoader) -> np.ndarray:
        """
        Generate class probability maps via patch-based inference.
        
        Args:
            data_loader: Yields (patch, coordinates) batches
            
        Returns:
            4D numpy array (6,Z,Y,X) of class probabilities
        """
        volume_shape = data_loader.dataset.volume.shape
        patch_size = data_loader.dataset.patch_size
        
        # Initialize accumulation buffers
        probabilities = torch.zeros((6, *volume_shape), dtype=torch.float32).to(self.device)
        counts = torch.zeros(volume_shape, dtype=torch.float32).to(self.device)
        
        # Weighting mask for smooth patch blending
        weight = torch.zeros(patch_size, dtype=torch.float32).to(self.device)
        weight[8:-8, 8:-8, 8:-8] = 1  # Full weight in center
        weight += 0.1  # Reduced weight in borders
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                for batch in data_loader:
                    batch["volume"] = batch["volume"].to(self.device)

                    patch_probs = self.model(batch)["particle"]  # (1,6,128,128,128)

                    for i, probs in enumerate(patch_probs):
                        z, y, x = batch["zyx"][i]
                        sl = (slice(z, z+patch_size[0]),
                            slice(y, y+patch_size[1]),
                            slice(x, x+patch_size[2]))
                        
                        counts[sl] += weight
                        probabilities[:, sl[0], sl[1], sl[2]] += probs * weight
        
        return (probabilities / counts).cpu().numpy()

    def _postprocess(self, probabilities: np.ndarray) -> DataFrame:
        """
        Convert probability maps to detected particle coordinates.
        
        Args:
            probabilities: 4D array (6,Z,Y,X) of class probs
            
        Returns:
            DataFrame with columns [x,y,z,particle_type]
        """
        results = []
        
        for particle_name, config in self.labels.items():
            label_idx = config["label"]
            threshold = config["threshold"]
            min_voxels = config["blob_threshold"]
            
            # Find connected components above threshold
            binary_map = probabilities[label_idx] > threshold
            labels = cc3d.connected_components(binary_map, connectivity=18)
            stats = cc3d.statistics(labels)
            
            # Filter by size and convert to coordinates
            stats["index"] = np.array(range(len(stats["voxel_counts"])))
            valid = (stats["voxel_counts"] > min_voxels) & (stats["index"] > 0)
            centroids = stats["centroids"][valid]
            
            if len(centroids) == 0:
                continue
            
            results.append(pd.DataFrame({
                "x": centroids[:, 2],
                "y": centroids[:, 1], 
                "z": centroids[:, 0],
                "particle_type": particle_name
            }))
        
        return pd.concat(results) if results else pd.DataFrame(columns=["x","y","z","particle_type"])

    def predict(self, context, model_input: np.ndarray) -> DataFrame:
        """
        End-to-end processing pipeline.
        
        Args:
            model_input: 3D numpy array (Z,Y,X) of input volume
            
        Returns:
            DataFrame of detected particles with coordinates and types
        """
        # 1. Prepare patches
        loader = self._preprocess(model_input)
        
        # 2. Run inference
        probs = self._get_probabilities(loader)
        
        # 3. Extract particles
        return self._postprocess(probs)