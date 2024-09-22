
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

# Load the DINO v2 model
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval().cuda()

# Define a transform to preprocess the images for DINO v2
transform = transforms.Compose([
    transforms.Resize(520),
    transforms.CenterCrop(518),  # should be multiple of model patch_size
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.2)
])


def extract_dino_features(image):
    """
    Extract DINO v2 features from a single image.
    
    Parameters:
        image (PIL.Image): Input image.
        
    Returns:
        torch.Tensor: Extracted features.
    """
    # Transform the image
    img_t = transform(image).unsqueeze(0).cuda()  # Transform and add batch dimension
    
    # Extract features
    with torch.no_grad():
        features_dict = dinov2_vitl14.forward_features(img_t)
        features = features_dict['x_norm_patchtokens']
        
    return features