from PIL import Image
from feature_pca import extract_pca_features


image_path = r"C:\Users\ykirc\Documents\GitHub\thesis\experimentation\input\frame_0001.png"
image = Image.open(image_path).convert('RGB')
features = extract_pca_features(image, n_components=1)
print("Features shape:", features.shape)