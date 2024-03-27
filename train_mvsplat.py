import random
import numpy as np

import hydra
from omegaconf import DictConfig

import torch
import torch.utils.data.distributed

from ggrt.global_cfg import set_cfg
from ggrt.model.mvsplat_network import MvsplatModel
from ggrt.visualization.feature_visualizer import *
from utils_loc import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr, data_shim
from train_ibrnet import synchronize
from ggrt.base.trainer import BaseTrainer

from ggrt.loss.criterion import MaskedL2ImageLoss
from math import ceil
# torch.autograd.set_detect_anomaly(True)
import copy
from torchvision.utils import save_image
from einops import rearrange



class MvSplatTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.state = 'nerf_only'

    def build_networks(self):
        self.model = MvsplatModel(self.config,
                                load_opt=not self.config.no_load_opt,
                                load_scheduler=not self.config.no_load_scheduler,
                                pretrained=self.config.pretrained)

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.gaussian_model.parameters(), lr=self.config.optimizer.lr)
        warm_up_steps = self.config.optimizer.warm_up_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                        1 / warm_up_steps,
                                                        1,
                                                        total_iters=warm_up_steps)

    def setup_loss_functions(self):
        self.rgb_loss = MaskedL2ImageLoss()


    def compose_state_dicts(self) -> None:
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict()}
        self.state_dicts['models']['gaussian'] = self.model.gaussian_model

    def train_iteration(self, batch) -> None:
        self.optimizer.zero_grad()
        if self.iteration == 0:
            self.state = self.model.switch_state_machine(state='nerf_only')
        batch = data_shim(batch, device=self.device)
        batch = self.model.gaussian_model.data_shim(batch)
        ret, data_gt = self.model.gaussian_model(batch, self.iteration)
        coarse_loss = self.rgb_loss(ret, data_gt)
        coarse_loss.backward()     

        self.optimizer.step()
        self.scheduler.step()
    
        if self.config.local_rank == 0 and self.iteration % self.config.n_tensorboard == 0:
            mse_error = img2mse(ret['rgb'], data_gt['rgb']).item(); psnr = mse2psnr(mse_error)
            self.scalars_to_log['train/coarse-loss'] = mse_error
            self.scalars_to_log['train/coarse-psnr'] = psnr
            self.scalars_to_log['lr/Gaussian'] = self.scheduler.get_last_lr()[0]
            print(f"train step: {self.iteration}; target: {int(batch['target']['index'][0])}; ref: {batch['context']['index']}; loss: {mse_error:.4f}, psnr: {psnr:.2f}")
        
    def validate(self) -> float:
        self.model.switch_to_eval()

        target_image = self.train_data['rgb'].squeeze(0).permute(2, 0, 1)
        self.writer.add_image('train/target_image', target_image, self.iteration)

        # Logging a random validation view.
        val_data = next(self.val_loader_iterator)
        score = 0
        # score = log_view_to_tb(
        #     self.writer, self.iteration, self.config, self.model,
        #     render_stride=self.config.render_stride, prefix='val/',
        #     data=val_data, dataset=self.val_dataset, device=self.device)
        torch.cuda.empty_cache()
        self.model.switch_to_train()

        return score



@torch.no_grad()
def log_view_to_tb(writer, global_step, args, model, render_stride=1, prefix='', data=None, dataset=None, device=None) -> float:

    print(f"validation step: {global_step}; target: {data['target']['index']}; ref: {data['context']['index']}")
    batch = data_shim(data, device=device)
    batch = model.gaussian_model.data_shim(batch)


    # features=  model.gaussian_model.encoder.backbone(batch['context'])
    # features = rearrange(features, "b v c h w -> b v h w c").to(torch.float)
    # features = model.gaussian_model.encoder.backbone_projection(features)
    # features = rearrange(features, "b v h w c -> b v c h w")
    ret, data_gt = model.gaussian_model(batch, global_step)
        

    average_im = batch['context']['image'].cpu().mean(dim=(0, 1))
    rgb_gt = data_gt['rgb'][0][0]
    rgb_pred = ret['rgb'][0][0].detach().cpu()

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['depth'].detach().cpu()[0][0]


    depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_im_gt_pred', rgb_im, global_step)
    writer.add_image(prefix + 'depth_pred', depth_im, global_step)

    # plot_feature_map(writer, global_step, ray_sampler, feat_maps, prefix)

    # write scalar
    # pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(rgb_pred, data_gt['rgb'][0][0].detach().cpu())
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    return psnr_curr_img

@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="finetune_mvsplat_stable",
)

def train(cfg_dict: DictConfig):
    args = cfg_dict
    set_cfg(cfg_dict)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Configuration for distributed training.
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        print(f'[INFO] Train in distributed mode')
        
    device = "cuda:{}".format(args.local_rank)

    trainer = MvSplatTrainer(args)
    trainer.train()

if __name__ == '__main__':
    train()
