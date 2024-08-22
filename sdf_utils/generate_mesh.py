import torch
import numpy as np
import mcubes
import trimesh
from tqdm import tqdm
import os
import math

def generate_mesh(model, N=512, return_sdf=False, save_dir=None, save_name=None):
    num_outputs = 1     # hard code this because the current models generate only one output
    # write output
    x = torch.linspace(-0.5, 0.5, N)
    if return_sdf:
        x = torch.arange(-N//2, N//2) / N
        x = x.float()
    x, y, z = torch.meshgrid(x, x, x)
    render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
    sdf_values = [np.zeros((N**3, 1)) for i in range(num_outputs)]

    # render in a batched fashion to save memory
    bsize = int(400**2)
    model.eval()
    for i in tqdm(range(math.ceil(N**3 / bsize))):
        coords = render_coords[i*bsize:(i+1)*bsize, :]
        with torch.no_grad():
            out = model(coords*2)

        if not isinstance(out, list):
            out = [out, ]

        for idx, sdf in enumerate(out):
            sdf_values[idx][i*bsize:(i+1)*bsize] = sdf.detach().cpu().numpy()

    for idx, sdf in enumerate(sdf_values):
        sdf = sdf.reshape(N, N, N)
        vertices, triangles = mcubes.marching_cubes(-sdf, 0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.vertices = (mesh.vertices / N - 0.5) + 0.5/N
    model.train()

    if return_sdf:
        return mesh, sdf
    else:
        return mesh