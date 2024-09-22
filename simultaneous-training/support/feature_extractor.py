import torch
import torchvision.transforms as T
from PIL import Image

# Load the DINO v2 model
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval().cuda()

# Define a transform to preprocess the images for DINO v2
transform = T.Compose([
    T.Resize(520),
    T.CenterCrop(518),  # should be multiple of model patch size
    T.ToTensor(),
    T.Normalize(mean=0.5, std=0.2)
])

def extract_features(image):
    """
    Extract features from an image using DINO v2 model.
    
    Parameters:
        image (PIL.Image): Input image in PIL format.
        
    Returns:
        np.ndarray: Extracted features reshaped to (batch_size, num_patches, feature_dim).
    """
    # Load and preprocess the image
    img_t = transform(image).unsqueeze(0).cuda()  # Transform and add batch dimension
    
    # Extract features using DINO v2 model
    with torch.no_grad():
        features_dict = dinov2_vitl14.forward_features(img_t)
        features = features_dict['x_norm_patchtokens']
    
    # Flatten the features to 2D and move to CPU
    batch_size, num_patches, feature_dim = features.shape
    features_flat = features.view(batch_size, num_patches, feature_dim).cpu().numpy()
    
    return features_flat


# input_image_path = "path/to/your/image.png"  # Replace with your image path
# image = Image.open(input_image_path).convert('RGB')
# features = extract_features(image)
# print("Features Shape:", features.shape)