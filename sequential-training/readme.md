# Sequential Training for Gaussian Models

This repository contains a comprehensive pipeline for training and clustering Gaussian models using DINO features. The project is designed to work on CUDA version 11.8 and has been tested on a Windows environment. Below, you will find a detailed guide on the structure of the repository, the purpose of each script, and the order in which to run them.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
    - [1. Convert Images](#1-convert-images)
    - [2. Extract DINO Features](#2-extract-dino-features)
    - [3. Train the Model](#3-train-the-model)
    - [4. Cluster the Model](#4-cluster-the-model)
5. [File Descriptions](#file-descriptions)
6. [Important Notes](#important-notes)
7. [Contact](#contact)

## Overview
This project aims to train and cluster Gaussian models using DINO features. The pipeline includes several steps:
1. **Image Conversion**: Convert images from a dataset into a format suitable for training.
2. **Feature Extraction**: Extract DINO features from the images.
3. **Model Training**: Train a Gaussian model using the extracted features.
4. **Clustering**: Cluster the trained Gaussian model for further analysis.

## Requirements
- Python 3.8 or higher
- CUDA 11.8
- Windows OS

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/sequential-training.git
    cd sequential-training
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure you have CUDA 11.8 installed and properly configured.

## Usage

### 1. Convert Images
The first step is to convert your images into a format suitable for training. This is done using the `convert.py` script.

```sh
python convert.py --source_path path/to/your/images --camera OPENCV --colmap_executable path/to/colmap --resize
```

### 2. Extract DINO Features
Next, extract DINO features from the converted images using the `dino-feature-extraction/extract.py` script.

```sh
python dino-feature-extraction/extract.py --dir_images path/to/converted/images --dir_dst path/to/save/features --model_path path/to/dino_model.pth
```

### 3. Train the Model
Once the features are extracted, you can train the Gaussian model using the `train.py` script.

```sh
python train.py --model_path path/to/save/model --source_path path/to/converted/images --iterations 25000
```

### 4. Cluster the Model
Finally, cluster the trained Gaussian model using the `cluster.py` script.

```sh
python cluster.py --model_path path/to/trained/model --save_path path/to/save/clusters --k 4
```

## File Descriptions

### `convert.py`
This script converts images from a dataset into a format suitable for training. It uses COLMAP for feature extraction, matching, and bundle adjustment, followed by image undistortion.

### `dino-feature-extraction/extract.py`
This script extracts DINO features from the images. It uses a pre-trained DINO model to extract features and applies PCA for dimensionality reduction.

### `train.py`
This script trains a Gaussian model using the extracted DINO features. It initializes the model, sets up the training parameters, and runs the training loop.

### `cluster.py`
This script clusters the trained Gaussian model. It uses K-means or Mean Shift clustering to group the Gaussian components and saves the results.

### `scene/`
This directory contains various utility scripts for handling scenes, cameras, and datasets.

### `utils/`
This directory contains utility scripts for general operations, image processing, and loss calculations.

## Important Notes
- **Custom Scene Training**: If you are training with a new scene, you must update the `dataset_readers.py` file. Specifically, change the path to the DINO features file on line 75:
    ```python
    dino_feats = torch.load("C:\\Users\\ykirc\\Desktop\\input-scenes\\1_snacks\\1_snacks.pt", weights_only=True)
    ```
    Update the path to point to your specific DINO features file.
