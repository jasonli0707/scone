import random 
import numpy as np
import torch
from dataset import *
import skimage

def psnr(img_true, img_test, data_range=1.):
    """ Compute Peak Signal to Noise Ratio metric """
    img_true = img_true.clip(0,1)
    img_test = img_test.clip(0,1)
    if isinstance(img_true, torch.Tensor):
        img_true = img_true.detach().cpu().numpy()
    if isinstance(img_test, torch.Tensor):
        img_test = img_test.detach().cpu().numpy()
    return skimage.metrics.peak_signal_noise_ratio(img_true, img_test, data_range=data_range)

def ssim(img1, img2, channel_axis=1, data_range=1.):
    """ Compute Structural Similarity Index Metric """
    img1 = img1.clip(0,1)
    img2 = img2.clip(0,1)
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    return skimage.metrics.structural_similarity(img1, img2, multichannel=True, channel_axis=channel_axis, data_range=data_range)

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_dataset(dataset_configs):
    if dataset_configs.data_type == "image":
        dataset = ImageFileDataset(dataset_configs)
    elif dataset_configs.data_type == "video":
        dataset = VideoDataset(dataset_configs)
    elif dataset_configs.data_type == "sdf":
        dataset = MeshSDF(dataset_configs)
    else:
        raise NotImplementedError(f"Dataset {dataset_configs.data_type} not implemented")
    return dataset

def get_model(model_configs, dataset):
    if model_configs.name == 'SIREN':
        from models.siren import Siren
        model = Siren(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            out_range=dataset.out_range,
            siren_configs=model_configs
        )
    elif model_configs.name == 'MFN':
        from models.mfn import MFNWrapper
        model = MFNWrapper(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            out_range=dataset.out_range,
            mfn_configs=model_configs.MFN_CONFIGS 
        )
    elif model_configs.name == 'SCONE':
        from models.scone import SCONE
        model = SCONE(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            scone_configs=model_configs
        )
    elif model_configs.name == "WIRE":
        from models.wire import Wire
        model = Wire(
           in_features=dataset.dim_in, 
           out_features=dataset.dim_out,
           wire_configs=model_configs
        )
    elif model_configs.name == "MLP":
        from models.mlp import MLP
        model = MLP(
            dim_in=dataset.dim_in,
            dim_out=dataset.dim_out,
            mlp_configs=model_configs
        )
    else:
        raise NotImplementedError(f"Model {model_configs.name} not implemented")
    return model

    
