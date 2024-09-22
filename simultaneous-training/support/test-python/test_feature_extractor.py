from PIL import Image
from feature_extractor import extract_features


image_path = r"C:\Users\ykirc\Documents\GitHub\thesis\experimentation\input\frame_0001.png"
image = Image.open(image_path).convert('RGB')
features = extract_features(image)
print("Features shape:", features.shape)