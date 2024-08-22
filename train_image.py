import os
import yaml
import shutil
import torch
import numpy as np
import pretty_errors
import hydra
import logging
from omegaconf import OmegaConf
from easydict import EasyDict
from tqdm import tqdm
from PIL import Image
from utils import *
from torch.utils.tensorboard import SummaryWriter

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

def save_src_for_reproduce(configs, out_dir):
    if os.path.exists(os.path.join('outputs', out_dir, 'src')):
        shutil.rmtree(os.path.join('outputs', out_dir, 'src'))
    shutil.copytree('models', os.path.join('outputs', out_dir, 'src', 'models'))
    # dump config to yaml file
    OmegaConf.save(dict(configs), os.path.join('outputs', out_dir, 'src', 'config.yaml'))

def save_image_and_tensorboard(out_dir, hwc_tensor, tag, color_mode='RGB', writer=None, step=None):
    """Save a HWC tensor as an image file."""
    img = hwc_tensor.clip(0, 1).detach().cpu().numpy()
    img = Image.fromarray((img * 255).astype(np.uint8), mode=color_mode)
    img.save(os.path.join('outputs', out_dir, f'{tag}.png'))

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
    H, W = dataset.H, dataset.W
    C = dataset.dim_out
    best_psnr, best_ssim = 0, 0
    best_pred = None

    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)
    ori_img = labels.view(H, W, C)
    ori_img = (ori_img + 1) / 2 if dataset.normalize else ori_img
    save_image_and_tensorboard(out_dir, ori_img, 'gt_img', writer=writer, step=0)    # save original image

    # train
    for step in process_bar:
        preds = model(coords, step, out_dir, return_loss=False, label=labels)
        loss = ((preds - labels) ** 2).mean()       # MSE loss

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        if dataset.normalize:
            # undo normalization [-1, 1] -> [0, 1]
            preds = (preds + 1) / 2
        preds = preds.clamp(0, 1)       # clip to [0, 1]
        preds = preds.view(H, W, C)
        psnr_score = psnr(ori_img, preds, data_range=1.0)
        ssim_score = ssim(preds, ori_img, channel_axis=2, data_range=1.0)
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('psnr', psnr_score, step)
        writer.add_scalar('ssim', ssim_score, step)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

        if psnr_score > best_psnr:
            best_psnr, best_ssim = psnr_score, ssim_score
            best_pred = preds

        # udpate progress bar
        process_bar.set_description(f"psnr: {psnr_score:.2f}, ssim: {ssim_score*100:.2f}, loss: {loss.item():.4f}")
        
        # save pred images        
        if step%train_configs.save_interval==0:
            print(f"Best psnr: {best_psnr:.2f}, ssim: {best_ssim*100:.2f}")
            log.info(f"Best psnr: {best_psnr:.2f}, ssim: {best_ssim*100:.2f}")
            save_image_and_tensorboard(out_dir, best_pred, 'best_pred', writer=writer, step=step)

    print("Training finished!")
    print(f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}")
    model_size = sum([p.numel() for p in model.parameters()])
    writer.add_text('metrics', f"Best psnr: {best_psnr:.4f}, ssim: {best_ssim*100:.4f}, model size: {model_size}", 0)
    writer.close()
    
    # save model
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'model.pth'))
    return best_psnr, best_ssim

@hydra.main(version_base=None, config_path='config', config_name='train_image')
def main(configs):
    seed_everything(42)
    device = "cuda"

    configs = EasyDict(configs)
    save_src_for_reproduce(configs, configs.TRAIN_CONFIGS.out_dir)

    # model and dataloader
    dataset = get_dataset(configs.DATASET_CONFIGS)
    model = get_model(configs.model_config, dataset)
    print(f"Start experiment: {configs.TRAIN_CONFIGS.out_dir}")
    n_params = sum([p.numel() for p in model.parameters()])
    print(f"No. of parameters: {n_params}")

    # train
    psnr, ssim = train(configs, model, dataset, device=device)
    log.info(f"Best PSNR: {psnr:.4f}, best SSIM: {ssim:.4f}")
    log.info(f"No. of parameters: {sum([p.numel() for p in model.parameters()])}")
    log.info(f"Results saved in: {configs.TRAIN_CONFIGS.out_dir}")
    return psnr, ssim, n_params

if __name__=='__main__':
    main()
