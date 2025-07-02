from typing import List
import numpy as np
from torch.utils.data import Dataset
import torch

def calculate_patch_starts(dimension_size: int, patch_size: int):
    if dimension_size <= patch_size:
        return [0]

    n_patches = np.ceil(dimension_size / patch_size)  +1 if dimension_size>300 else np.ceil(dimension_size / patch_size) 

    if n_patches == 1:
        return [0]
    
    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)
        
    return positions

class PatchDataset(Dataset):
    def __init__(self,volume: np.ndarray, patch_size: List[int]):

        self.patch_size = patch_size
        self.volume = volume
        self.indexes = [[z,y,x] 
                       for z in calculate_patch_starts(volume.shape[0], patch_size[0])
                       for y in calculate_patch_starts(volume.shape[1], patch_size[1])
                       for x in calculate_patch_starts(volume.shape[2], patch_size[2])]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):

        zyx = self.indexes [idx]
        patch = self.volume[zyx[0]:zyx[0]+self.patch_size[0],zyx[1]:zyx[1]+self.patch_size[1],zyx[2]:zyx[2]+self.patch_size[2]]

        return {
            "volume": patch,
            "zyx": torch.tensor(zyx,dtype = torch.long)
        }