import os
import yaml
import shutil
import torch
import numpy as np
import pretty_errors
import hydra
from omegaconf import OmegaConf
from easydict import EasyDict
from tqdm import tqdm
from utils import seed_everything, get_dataset, get_model
from torch.utils.tensorboard import SummaryWriter
from utils import psnr, ssim
import cv2

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

def load_config(config_file):
    configs = yaml.safe_load(open(config_file))
    return configs

def save_src_for_reproduce(configs, out_dir):
    if os.path.exists(os.path.join('outputs', out_dir, 'src')):
        shutil.rmtree(os.path.join('outputs', out_dir, 'src'))
    shutil.copytree('models', os.path.join('outputs', out_dir, 'src', 'models'))
    # dump config to yaml file
    OmegaConf.save(dict(configs), os.path.join('outputs', out_dir, 'src', 'config.yaml'))


def save_checkpoint(model, optimizer, scheduler, step, out_dir):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step
    }
    torch.save(checkpoint, os.path.join('outputs', out_dir, 'best_model.pth'))


def train(configs, model, dataset, device='cuda'):
    train_configs = configs.TRAIN_CONFIGS
    dataset_configs = configs.DATASET_CONFIGS
    out_dir = train_configs.out_dir
    batch_size = dataset_configs.batch_size
    # tensorboard
    writer = SummaryWriter(os.path.join('outputs', out_dir, 'tensorboard'))

    # optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=train_configs.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_configs.iterations, eta_min=1e-6)

    # prepare training settings
    model = model.to(device)
    process_bar = tqdm(range(train_configs.iterations))
    T, H, W, C = dataset.T, dataset.H, dataset.W, dataset.C
    best_psnr = 0
    best_psnr_std = 0
    best_ssim = 0
    best_ssim_std = 0

    coords, labels = dataset.get_data()
    coords, labels = coords.to(device), labels.to(device)

    # train
    for step in process_bar:
        model.train()
        idx = torch.randint(0, len(coords), (batch_size,)) # random sample points
        coords_batch, labels_batch = coords[idx], labels[idx]
        preds = model(coords_batch, step, out_dir)
        loss = ((preds - labels_batch) ** 2).mean()       # MSE loss

        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        psnr_score = -10*np.log10(loss.item()) + 20*np.log10(2) # data_range=2
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('psnr', psnr_score, step)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], step)


        # udpate progress bar
        process_bar.set_description(f"train psnr: {psnr_score:.2f}, train loss: {loss.item():.4f}")

        if step%train_configs.val_interval==0 and step:
            model.eval()
            with torch.no_grad(): # predict and compute psnr frame by frame
                coords_frames, label_frames = coords.reshape(T, H, W, C), labels.reshape(T, H, W, C)
                label_frames = ((label_frames+1)*0.5).cpu().numpy() # unnormalize
                psnr_frames = []
                ssim_frames = []
                for i in range(T):
                    pred_frame = model(coords_frames[i].reshape(-1, 3), step, out_dir)
                    pred_frame = pred_frame.reshape(H, W, C)
                    pred_frame = ((pred_frame+1)*0.5).detach().cpu().numpy() # unnormalize
                    pred_frame = np.clip(pred_frame, 0, 1)
                    psnr_frames.append(psnr(pred_frame, label_frames[i]))
                    ssim_frames.append(ssim(pred_frame, label_frames[i], channel_axis=2))

            val_psnr_mean, val_psnr_std = np.mean(psnr_frames), np.std(psnr_frames)
            val_ssim_mean, val_ssim_std = np.mean(ssim_frames), np.std(ssim_frames)
            writer.add_scalar('val_psnr_mean', val_psnr_mean, step) 
            writer.add_scalar('val_psnr_std', val_psnr_std, step) 
            writer.add_scalar('val_ssim_mean', val_ssim_mean, step) 
            writer.add_scalar('val_ssim_std', val_ssim_std, step) 
            if val_psnr_mean > best_psnr:
                best_psnr = val_psnr_mean
                best_psnr_std = val_psnr_std
                best_ssim = val_ssim_mean
                best_ssim_std = val_ssim_std
                print(f"Best val psnr: {best_psnr:.2f} +/- {best_psnr_std:.2f}")
                print(f"Best val ssim: {best_ssim:.2f} +/- {best_ssim_std:.2f}")
                save_checkpoint(model, opt, scheduler, step, out_dir)
        

    print("Training finished!")
    print(f"Best val psnr: {best_psnr:.4f}, std: {best_psnr_std:.4f}")
    model_size = sum([p.numel() for p in model.parameters()])
    writer.add_text('metrics', f"Best val psnr: {best_psnr:.4f}, std: {best_psnr_std:.4f} , model size: {model_size}", 0)
    writer.close()
    
    # save last checkpoint
    torch.save(model.state_dict(), os.path.join('outputs', out_dir, 'last_model.pth'))
    return best_psnr, best_ssim

@hydra.main(version_base=None, config_path='config', config_name='train_video')
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
    return psnr, ssim, n_params

if __name__=='__main__':
    main()
