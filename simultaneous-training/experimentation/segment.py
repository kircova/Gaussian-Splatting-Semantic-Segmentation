import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import json
import glob
from tqdm.notebook import tqdm
from torchvision import transforms
import sys
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


def save_pca_image(pca_features, patch_h, patch_w, output_image_path):
    """
    Save the PCA-transformed features as an image.
    
    Parameters:
        pca_features (np.ndarray): PCA-transformed features.
        patch_h (int): Height of the patch grid.
        patch_w (int): Width of the patch grid.
        output_image_path (str): Path to save the output image.
    """
    single_component = pca_features.squeeze()  # Remove batch dimension if present
    single_component = single_component.reshape(patch_h, patch_w)
    
    # Scale to 0-255 and convert to uint8
    single_component = (single_component * 255).astype(np.uint8)
    
    # Save the image
    Image.fromarray(single_component).save(output_image_path)
    print(f"PCA image saved to {output_image_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python segment_image.py <input_image_path> <output_image_path>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    image = Image.open(input_image_path).convert('RGB')

    pca_features = extract_pca_features(image, 1)
    print("PCA Features Shape:", pca_features.shape)

    patch_size = dinov2_vitl14.patch_size
    patch_h, patch_w = 518 // patch_size, 518 // patch_size

    # Save PCA features as an image
    save_pca_image(pca_features, patch_h, patch_w, output_image_path)

#input_image_path = "input/frame_0001.png"  # Replace with your image path
#image = Image.open(input_image_path).convert('RGB')

#pca_features = extract_pca_features(input_image_path, n_components=1)

#print("PCA Features Shape:", pca_features.shape)