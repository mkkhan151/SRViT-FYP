# Super Resolution Vision Transformer (SRViT)


## Overview

Super Resolution Vision Transformer (SRViT) is a deep learning project that leverages Vision Transformer (ViT) architectures for single image super-resolution tasks. The goal is to reconstruct high-resolution images from their low-resolution counterparts using transformer-based models.

## Features

- Utilizes Vision Transformer (ViT) for image super-resolution
- Supports multiple upscaling factors (e.g., 2x, 4x)
- Modular and extensible codebase
- Training and evaluation notebooks included
- Compatible with popular datasets (e.g., DIV2K, Set5)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mkkhan151/SRViT-FYP.git
    cd SRViT-FYP
    ```

2. Create and activate a virtual environment (Recommended):
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
- For Training, visit [SRViT_Training_x2](notebooks/SRViT_Training_x2.ipynb) in notebooks folder.
- It contains all code data loading, model architecture, training, and evaluation.

### Evaluation
- visit the [SRViT_Training_x2](notebooks/SRViT_Training_x2.ipynb) for detailed instructions.

### Streamlit App
- To test the model, run streamlit app after creating virtual environment.
  ```bash
  streamlit run app.py
  ```