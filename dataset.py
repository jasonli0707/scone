import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from einops import rearrange
import numpy as np
import os

class ImageFileDataset(Dataset):
    def __init__(self, dataset_configs):
        super().__init__()
        self.config = dataset_configs
        self.normalize = dataset_configs.normalize

        # load image and convert to tensor
        img = Image.open(dataset_configs.img_file)
        if img.mode=='RGBA':
            img = img.convert('RGB')
        if dataset_configs.color_mode == 'YCbCr':
            # convert to YCbCr
            img = img.convert('YCbCr')
        self.img = img
        self.img_size = img.size
        
        img_tensor = ToTensor()(img)
        if self.normalize:
            img_tensor = img_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        img_tensor = rearrange(img_tensor, 'c h w -> (h w) c')
        self.labels = img_tensor

        # build coords
        W, H = self.img_size
        grid = [torch.linspace(-1., 1., H), torch.linspace(-1., 1., W)] # normalized coords
        self.coords = torch.stack(
            torch.meshgrid(grid),
            dim=-1,
        ).view(-1, 2)

        # set x, y dimensions
        self.H, self.W = H, W
        self.dim_in = 2
        self.dim_out = 3
        self.out_range = (-1.0, 1.0) if self.normalize else (0.0, 1.0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_data(self):
        return self.coords, self.labels


class VideoDataset(Dataset):
    def __init__(self, dataset_configs):
        super().__init__()
        self.config = dataset_configs
        self.normalize = dataset_configs.normalize

        self.video = torch.tensor(np.load(dataset_configs.video_file)) # video_file as numpy array [T, H, W, C]
        if self.normalize:
            self.video = 2*self.video - 1. # normalize to [-1, 1]
        self.T, self.H, self.W, self.C = self.video.shape
        mesh_sizes = self.video.shape[:-1]
        self.coords = torch.stack(
            torch.meshgrid([torch.linspace(-1., 1., s) for s in mesh_sizes]), dim=-1
        ).view(-1, 3)  # normalized coords
        self.video = self.video.view(-1, self.C)

        # set x, y, t dimensions
        self.dim_in = 3
        self.dim_out = 3
        self.out_range = (-1.0, 1.0) if self.normalize else (0.0, 1.0)

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return self.coords[idx], self.video[idx]

    def get_data(self):
        return self.coords, self.video

class MeshSDF(Dataset):
    ''' convert point cloud to SDF '''

    def __init__(self, configs):
        super().__init__()
        self.num_samples = configs.num_samples
        self.pointcloud_path = configs.xyz_file
        self.coarse_scale = configs.coarse_scale
        self.fine_scale = configs.fine_scale
        self.normalize = True
        self.dim_in = 3
        self.dim_out = 1
        self.out_range = None

        # load gt point cloud with normals
        self.load_mesh(configs.xyz_file)
        
        # precompute sdf and occupancy grid
        render_resolution = configs.render_resolution
        self.render_resolution = render_resolution
        self.load_precomputed_occu_grid(configs.xyz_file, render_resolution)

    def load_precomputed_occu_grid(self, xyz_file, render_resolution):
        # load from files if exists
        sdf_file = xyz_file.replace('.xyz', f'_{render_resolution}_sdf.npy')
        if os.path.exists(sdf_file):
            self.sdf = np.load(sdf_file)
        else:
            self.sdf = self.build_sdf(render_resolution)
            np.save(sdf_file, self.sdf)

        occu_grid = (self.sdf <= 0)
        self.occu_grid = occu_grid

    def build_sdf(self, render_resolution):
        N = render_resolution
        # build grid
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
        x, y, z = torch.meshgrid(x, x, x)
        render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
        vox_centers = render_coords.cpu().numpy()

        # use KDTree to get nearest neighbours and estimate the normal
        _, idx = self.kd_tree.query(vox_centers, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((vox_centers - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf.reshape(N, N, N)
        return sdf
    
    def build_grid_coords(self, render_resolution):
        N = render_resolution
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
        x, y, z = torch.meshgrid(x, x, x)
        coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
        return coords.cpu().numpy()

    def load_mesh(self, pointcloud_path):
        from pykdtree.kdtree import KDTree
        npy_file = pointcloud_path.replace('.xyz', '.npy')
        if os.path.exists(npy_file):
            pointcloud = np.load(npy_file)
        else:
            pointcloud = np.genfromtxt(pointcloud_path)
            np.save(pointcloud_path.replace('.xyz', '.npy'), pointcloud)
        self.pointcloud = pointcloud
        
        # cache to speed up loading
        self.v = pointcloud[:, :3]
        self.n = pointcloud[:, 3:]

        n_norm = (np.linalg.norm(self.n, axis=-1)[:, None])
        n_norm[n_norm == 0] = 1.
        self.n = self.n / n_norm
        self.v = self.normalize_coords(self.v)
        self.kd_tree = KDTree(self.v)
        print('finish loading pc')

    def normalize_coords(self, coords):
        coords -= np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def sample_surface(self):
        idx = np.random.randint(0, self.v.shape[0], self.num_samples)
        points = self.v[idx]
        points[::2] += np.random.laplace(scale=self.coarse_scale, size=(points.shape[0]//2, points.shape[-1]))
        points[1::2] += np.random.laplace(scale=self.fine_scale, size=(points.shape[0]//2, points.shape[-1]))

        # wrap around any points that are sampled out of bounds
        points[points > 0.5] -= 1
        points[points < -0.5] += 1

        # use KDTree to get distance to surface and estimate the normal
        sdf, idx = self.kd_tree.query(points, k=3)
        avg_normal = np.mean(self.n[idx], axis=1)
        sdf = np.sum((points - self.v[idx][:, 0]) * avg_normal, axis=-1)
        sdf = sdf[..., None]

        return points, sdf

    def __getitem__(self, idx):
        batch_size = 262144
        start_idx = idx * batch_size
        coords, sdf = self.get_data(start_idx, batch_size)
        return coords, sdf
    
    def __len__(self):
        return 1
    
    def get_data(self):
        coords, sdf = self.sample_surface()
        return torch.from_numpy(coords).float(), torch.from_numpy(sdf).float()