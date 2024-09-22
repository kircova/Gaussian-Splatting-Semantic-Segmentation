import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import transforms
from feature_pca import extract_pca_features

def visualize_segmentation(pca_features, original_size, patch_h, patch_w, threshold=0.35):
    """
    Segment and visualize the background and foreground using the first PCA component.
    
    Parameters:
        pca_features (np.ndarray): PCA-transformed features.
        original_size (tuple): Original size of the video frames (height, width).
        patch_h (int): Height of the patch grid.
        patch_w (int): Width of the patch grid.
        threshold (float): Threshold value to separate background and foreground.
    """
    # Extract the first PCA component and reshape
    pca_first_component = pca_features[:, 0].reshape(-1, patch_h, patch_w)
    
    # Segment the background and foreground
    pca_features_bg = pca_first_component > threshold  # Background mask
    pca_features_fg = ~pca_features_bg  # Foreground mask
    
    visualizations = []
    for i in range(pca_features_bg.shape[0]):
        bg_resized = cv2.resize(pca_features_bg[i].astype(np.uint8) * 255, original_size[::-1])
        fg_resized = cv2.resize(pca_features_fg[i].astype(np.uint8) * 255, original_size[::-1])
        
        # Stack the masks into 3 channels to visualize in color
        bg_colored = cv2.applyColorMap(bg_resized, cv2.COLORMAP_VIRIDIS)
        fg_colored = cv2.applyColorMap(fg_resized, cv2.COLORMAP_VIRIDIS)
        
        combined = cv2.addWeighted(bg_colored, 0.5, fg_colored, 0.5, 0)
        visualizations.append(combined)
    
    visualizations = np.array(visualizations)
    return visualizations

def process_video(input_video_path, output_video_path, n_components=1, threshold=0.35):
    """
    Process a video file, segment background and foreground, and save the result as a new video.
    
    Parameters:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        n_components (int): Number of PCA components.
        threshold (float): Threshold value to separate background and foreground.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i}")
            break

        # Convert the frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Extract PCA features
        pca_features = extract_pca_features(pil_image, n_components)

        # Determine patch grid size
        num_patches = pca_features.shape[1]
        patch_h, patch_w = int(np.sqrt(num_patches)), int(np.sqrt(num_patches))
        
        # Visualize segmentation with background and foreground separation
        vis_frame = visualize_segmentation(pca_features, (height, width), patch_h, patch_w, threshold).squeeze()

        # Write the frame to the output video
        out.write(vis_frame)

        print(f"Processed frame {i+1}/{frame_count}")

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved as {output_video_path}")

# Example usage:
input_video_path = r"C:\Users\ykirc\Documents\GitHub\thesis\support\test-media\img4.mp4"  # Replace with your input video path
output_video_path = "output_video_segmented.mp4"  # Replace with your desired output video path

process_video(input_video_path, output_video_path, n_components=1, threshold=0.35)
