**🏆 MLflow Model Packaging Example (5th Place — CZII CryoET Object Identification)**

This repository contains the packaged version of the model that ranked 5th place in the [CZII - CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification)

**📂 Project Structure**
├── model_code 
│├── mlflow_model.py # MLflow wrapper for the PyTorch model 
│├── patch_dataset.py # PyTorch Patch Dataset and DataLoader for Inference 
│ └── pytorch_model.py # Defines the model architecture 
├── model_data 
│ ├── labels.txt # Auxiliary file for output transformation 
│└── model.pt # Trained PyTorch model 
├── package.py # Script to package model into MLflow format
├── requirements.txt # Python dependencies 
└── quickstart.ipynb # Notebook to run the example end-to-end

**⚡ Quickstart**

- To get started quickly, open and run quickstart.ipynb in Colab Notebook or VSCode. It walks you through:
- Setting up the environment
- Packaging the model with MLflow
- Loading and testing the packaged model

**Citation**

```bibtex
@misc{czii-cryo-et-object-identification,
    author = {Kyle Harrington* and Mohammadreza Paraan* and Anchi Cheng and Utz Heinrich Ermel and Saugat Kandel and Dari Kimanius and Elizabeth Montabana and Ariana Peck and Jonathan Schwartz and Daniel Serwas and Hannah Siems and Feng Wang and Yue Yu and Zhuowen Zhao and Shawn Zheng and Walter Reade and Maggie Demkin and Kristen Maitland and Dannielle McCarthy and Matthias Haury and David Agard and Clinton Potter and Bridget Carragher},
    title = {CZII - CryoET Object Identification},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/czii-cryo-et-object-identification}},
    note = {Kaggle}
}