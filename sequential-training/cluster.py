import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
import open3d as o3d

# Note: You'll need to import these from your project
from scene import GaussianModel
from gaussian_renderer import render
from arguments import PipelineParams

class DinoModel:
    def __init__(self, model_path):
        self.load_model(model_path)

    def load_model(self, model_path):
        print("\n[Loading the gaussian model]")
        saved_dict = torch.load(model_path)

        self.gaussians = GaussianModel(0)
        self.gaussians.active_sh_degree = saved_dict['active_sh_degree']
        self.gaussians.max_sh_degree = saved_dict['max_sh_degree']
        self.gaussians._xyz = nn.Parameter(saved_dict['xyz'])
        self.gaussians._features_dc = nn.Parameter(saved_dict['features_dc'])
        self.gaussians._features_rest = nn.Parameter(saved_dict['features_rest'])
        self.gaussians._scaling = nn.Parameter(saved_dict['scaling'])
        self.gaussians._rotation = nn.Parameter(saved_dict['rotation'])
        self.gaussians._opacity = nn.Parameter(saved_dict['opacity'])
        self.gaussians._feature_image1 = nn.Parameter(saved_dict['feature_image1'])
        self.gaussians._feature_image2 = nn.Parameter(saved_dict['feature_image2'])
        self.gaussians._feature_image3 = nn.Parameter(saved_dict['feature_image3'])
        self.gaussians._feature_image4 = nn.Parameter(saved_dict['feature_image4'])
        self.gaussians._feature_images = [self.gaussians._feature_image1, self.gaussians._feature_image2, 
                                      self.gaussians._feature_image3, self.gaussians._feature_image4]
        self.cameras = saved_dict['cameras']

        print("[Gaussian model loaded successfully]")

    def construct_list_of_attributes(self):
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attributes += [f'f_dc_{i}' for i in range(self.gaussians._features_dc.shape[1] * self.gaussians._features_dc.shape[2])]
        attributes += [f'f_rest_{i}' for i in range(self.gaussians._features_rest.shape[1] * self.gaussians._features_rest.shape[2])]
        attributes += ['opacity']
        attributes += [f'scale_{i}' for i in range(self.gaussians._scaling.shape[1])]
        attributes += [f'rot_{i}' for i in range(self.gaussians._rotation.shape[1])]
        return attributes

    def get_dino_features(self):
        feature_images = torch.cat(self.gaussians._feature_images, dim=1)
        return feature_images.reshape(-1, feature_images.shape[1]).cpu().detach().numpy()

    def get_point_cloud_data(self):
        xyz = self.gaussians._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.gaussians._opacity.detach().cpu().numpy()
        scale = self.gaussians._scaling.detach().cpu().numpy()
        rotation = self.gaussians._rotation.detach().cpu().numpy()
        return xyz, normals, f_dc, f_rest, opacities, scale, rotation

def cluster_and_save(model, save_path, n_clusters=None, use_mean_shift=False):
    feature_images = model.get_dino_features()
    
    if use_mean_shift:
        bandwidth = estimate_bandwidth(feature_images, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        labels = ms.fit_predict(feature_images)
        n_clusters = len(np.unique(labels))
        print(f"Mean Shift clustering completed. Number of clusters: {n_clusters}")
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(feature_images)
        print(f"K-means clustering completed. Number of clusters: {n_clusters}")

    xyz, normals, f_dc, f_rest, opacities, scale, rotation = model.get_point_cloud_data()
    attributes = model.construct_list_of_attributes()

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        save_cluster(i, cluster_indices, xyz, normals, f_dc, f_rest, opacities, scale, rotation, attributes, save_path)

    return labels

def save_cluster(cluster_id, indices, xyz, normals, f_dc, f_rest, opacities, scale, rotation, attributes, save_path):
    cluster_data = np.column_stack((
        xyz[indices], normals[indices], f_dc[indices], f_rest[indices],
        opacities[indices], scale[indices], rotation[indices]
    ))
    
    dtype_full = [(attr, 'f4') for attr in attributes]
    elements = np.empty(len(indices), dtype=dtype_full)
    elements[:] = list(map(tuple, cluster_data))
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(os.path.join(save_path, f'cluster_{cluster_id}.ply'))
    print(f"Saved cluster_{cluster_id}.ply")

def extract_gaussian_attributes(ply_path):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']
    
    attributes = {}
    for prop in vertex_data.properties:
        attributes[prop.name] = vertex_data[prop.name]
    
    return attributes

def create_masked_ply(original_attributes, mask_color, blend_ratio=0.5):
    new_attributes = original_attributes.copy()
    
    # Apply mask to f_dc (assuming these are the main color components)
    for i in range(3):
        f_dc = original_attributes[f'f_dc_{i}']
        masked_f_dc = f_dc * (1 - blend_ratio) + mask_color[i] * blend_ratio
        new_attributes[f'f_dc_{i}'] = masked_f_dc
    
    return new_attributes

def save_ply_with_attributes(filename, attributes):
    dtype_list = [(name, 'f4') for name in attributes.keys()]
    vertex_data = np.empty(len(attributes['x']), dtype=dtype_list)
    
    for name in attributes:
        vertex_data[name] = attributes[name]
    
    el = PlyElement.describe(vertex_data, 'vertex')
    PlyData([el]).write(filename)

def combine_clusters(input_path, output_path):
    class_colors = [
        np.array([1, 0, 0]),       # Red
        np.array([0, 1, 0]),       # Green
        np.array([0, 0, 1]),       # Blue
        np.array([1, 0, 1]),       # Magenta
        np.array([1, 1, 0]),       # Yellow
        np.array([0, 1, 1]),       # Cyan
        np.array([0.5, 0, 0]),     # Dark Red
        np.array([0, 0.5, 0]),     # Dark Green
        #np.array([0, 0, 0.5]),     # Dark Blue
        #np.array([0.5, 0.5, 0]),   # Olive
        #np.array([0.5, 0, 0.5]),   # Purple
        #np.array([0, 0.5, 0.5]),   # Teal
        #np.array([1, 0.5, 0]),     # Orange
        #np.array([1, 0, 0.5]),     # Rose
        #np.array([0.5, 1, 0]),     # Lime
        #np.array([0, 1, 0.5]),     # Spring Green
        #np.array([0.5, 0, 1]),     # Violet
        #np.array([0, 0.5, 1]),     # Azure
        #np.array([1, 0.75, 0.8]),  # Pink
        #np.array([0.8, 1, 0.6]),   # Light Green
    ]

    print("Loading point clouds...")
    all_attributes = []
    cluster_files = [f for f in os.listdir(input_path) if f.startswith("cluster_") and f.endswith(".ply")]
    
    for i, filename in enumerate(cluster_files):
        attributes = extract_gaussian_attributes(os.path.join(input_path, filename))
        all_attributes.append(attributes)
        print(f"Loaded {filename} with {len(attributes['x'])} points")

    # Create masked attributes for each cluster
    masked_attributes = []
    for i, attributes in enumerate(all_attributes):
        masked = create_masked_ply(attributes, class_colors[i % len(class_colors)], 0.9)
        masked_attributes.append(masked)
        save_ply_with_attributes(os.path.join(input_path, f"masked_cluster_{i}.ply"), masked)
        print(f"Saved masked_cluster_{i}.ply")

    # Combine all point clouds
    combined_attributes = {k: np.concatenate([attr[k] for attr in masked_attributes]) for k in masked_attributes[0].keys()}

    # Save the combined point cloud
    save_ply_with_attributes(output_path, combined_attributes)
    print(f"Successfully saved combined point cloud to {output_path}")

    # Load and visualize the combined point cloud using Open3D
    combined_pcd = o3d.io.read_point_cloud(output_path)
    o3d.visualization.draw_geometries([combined_pcd])

def main(args):
    print(f"Processing Gaussian model: {args.model_path}")

    # Create results directory
    os.makedirs(args.save_path, exist_ok=True)

    # Load the gaussian model
    model = DinoModel(args.model_path)

    # Perform clustering
    labels = cluster_and_save(model, args.save_path, args.k, args.mean_shift)

    # Combine and visualize clusters
    combined_path = os.path.join(args.save_path, 'combined_clusters.ply')
    combine_clusters(args.save_path, combined_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clustering gaussians with DINO features and visualizing results')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the clustering results')
    parser.add_argument('--k', type=int, default=4, help='Number of clusters for K-means')
    parser.add_argument('--mean_shift', action='store_true', help='Use Mean Shift clustering instead of K-means')
    args = parser.parse_args()

    main(args)