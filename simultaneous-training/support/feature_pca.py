import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms

# Load the DINO v2 model
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval().cuda()

# Define a transform to preprocess the images for DINO v2
transform = transforms.Compose([
    transforms.Resize(520),
    transforms.CenterCrop(518),  # should be multiple of model patch_size
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.2)
])

def extract_pca_features(image, n_components=1):
    """
    Extract PCA-transformed features from an image using DINO v2 model.
    
    Parameters:
        image_path (str): Path to the input image.
        n_components (int): Number of PCA components.
        
    Returns:
        np.ndarray: PCA-transformed features.
    """
    # Load and preprocess the image
    img_t = transform(image).unsqueeze(0).cuda()  # Transform and add batch dimension
    
    # Extract features using DINO v2 model
    with torch.no_grad():
        features_dict = dinov2_vitl14.forward_features(img_t)
        features = features_dict['x_norm_patchtokens']
    
    # Flatten the features to 2D
    batch_size, num_patches, feature_dim = features.shape
    features_flat = features.view(-1, feature_dim).cpu().numpy()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_flat)
    
    # Normalize PCA features to [0, 1] range
    pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
    
    return pca_features.reshape(batch_size, num_patches, n_components)


#input_image_path = "input/frame_0001.png"  # Replace with your image path
#image = Image.open(input_image_path).convert('RGB')

#pca_features = extract_pca_features(input_image_path, n_components=1)

#print("PCA Features Shape:", pca_features.shape)