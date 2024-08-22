import sys
import os
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import pretty_errors
import hydra
import logging
from tqdm import tqdm
import yaml
from utils import seed_everything, get_dataset, get_model
from sdf_utils.generate_mesh import generate_mesh
from sdf_utils import open3d_utils
import mcubes
import trimesh

# config pretty_errors
pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_EXTENDED,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    truncate_code       = True,
    display_locals      = True,
)

log = logging.getLogger(__name__)

def load_config(config_file):
    configs = yaml.safe_load(open(config_file))
    return configs

def train(configs, model, dataset, device='cuda'):
    train_configs = configs.TRAIN_CONFIGS
    out_dir = train_configs.out_dir
    # tensorboard
    writer = SummaryWriter(os.path.join('outputs', out_dir, 'tensorboard'))

    # optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)

    # prepare training settings
    model.train()
    model = model.to(device)
    process_bar = tqdm(range(train_configs.iterations))
    C = dataset.dim_out
    best_iou, best_loss, best_cd = 0, 1e10, 1e10
    best_mesh = None

    # train
    for step in process_bar:
        coords, labels = dataset.get_data()
        coords, labels = coords.to(device), labels.to(device)    
        # check any value is nan
        assert not torch.isnan(coords).any() and not torch.isnan(labels).any(), "nan value detected!"
        preds = model(coords*2, step, out_dir)
        assert not torch.isnan(preds).any(), "nan value detected!"
        loss = ((preds - labels) ** 2).mean()       # MSE loss

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # scores
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

        if loss.item() < best_loss:
            best_loss = loss.item()
            
        if (step%5000==0 and step>0) or (step==train_configs.iterations-1):
            mesh, pred_sdf = generate_mesh(model, N=dataset.render_resolution ,return_sdf=True, save_dir=os.path.join('outputs', out_dir, 'meshes'), save_name=f"best_mesh")
            pred_occ = pred_sdf <= 0
            gt_occ = dataset.occu_grid
            intersection = np.sum(np.logical_and(gt_occ, pred_occ))
            union = np.sum(np.logical_or(gt_occ, pred_occ))
            iou = intersection / union

            # compute chamfer distance
            sdf = dataset.sdf
            vertices, triangles = mcubes.marching_cubes(-sdf, 0)
            N = dataset.render_resolution
            gt_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            gt_mesh.vertices = (gt_mesh.vertices / N - 0.5) + 0.5/N
            pc_gt = open3d_utils.create_o3d_pointcloud(gt_mesh.vertices)
            pc_pred = open3d_utils.create_o3d_pointcloud(mesh.vertices)
            chamfer_dist1 = pc_pred.compute_point_cloud_distance(pc_gt)
            chamfer_dist2 = pc_gt.compute_point_cloud_distance(pc_pred)
            chamfer_dist = np.asarray(chamfer_dist1).mean() + np.asarray(chamfer_dist2).mean()
            print(f"chamfer distance: {chamfer_dist}, iou: {iou}")

            if iou > best_iou:
                best_cd = chamfer_dist
                best_iou = iou
                best_mesh = mesh

        # udpate progress bar
        process_bar.set_description(f"loss x 10000: {loss.item()*10000:.4f}, best_iou: {best_iou*100:.2f}")
        
    # save mesh
    o3d_mesh = open3d_utils.trimesh_to_o3d_mesh(best_mesh)
    os.makedirs(os.path.join('outputs', out_dir, 'meshes'), exist_ok=True)
    open3d_utils.save_mesh(o3d_mesh, os.path.join('outputs', out_dir, 'meshes', 'best_mesh.ply'))

    print("Training finished!")
    print(f"Best iou: {best_iou:.4f}, best cd: {best_cd:.4f}")
    log.info(f"Best iou: {best_iou:.4f}, best cd: {best_cd:.4f}")
    model_size = sum([p.numel() for p in model.parameters()])
    writer.add_text('metrics', f"Best iou: {best_iou:.4f}, best cd: {best_cd:.4f}, model size: {model_size}", 0)
    writer.close()
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))
    return best_iou, best_loss, best_cd, model_size

@hydra.main(version_base=None, config_path='config', config_name='train_sdf')
def main(configs):
    seed_everything(42)
    device = "cuda"

    configs = EasyDict(configs)

    # model and dataloader
    dataset = get_dataset(configs.DATASET_CONFIGS)
    model = get_model(configs.model_config, dataset)
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")
    n_params = sum([p.numel() for p in model.parameters()])
    print(f"No. of parameters: {n_params}")

    # train
    best_iou, best_loss, best_cd, model_size = train(configs, model, dataset, device=device)
    log.info(f"Best iou: {best_iou:.4f}, best cd: {best_cd:.4f}, best loss: {best_loss:.4f}")
    log.info(f"No. of parameters: {sum([p.numel() for p in model.parameters()])}")
    log.info(f"Results saved in: {configs.TRAIN_CONFIGS.out_dir}")
    return best_iou, best_cd

if __name__=='__main__':
    main()
