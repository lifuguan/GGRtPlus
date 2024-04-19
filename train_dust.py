import random
import numpy as np
import hydra
from omegaconf import DictConfig

import torch
import torch.utils.data.distributed

from ggrt.global_cfg import set_cfg
from ggrt.geometry.depth import inv2depth
from ggrt.geometry.align_poses import align_ate_c2b_use_a2b
from ggrt.model.dbarf import DBARFModel
from ggrt.model.dust_gs import dust_gs
from ggrt.projection import Projector
from ggrt.pose_util import Pose, rotation_distance
from ggrt.render_ray import render_rays
from ggrt.render_image import render_single_image
from ggrt.sample_ray import RaySamplerSingleImage
from ggrt.visualization.pose_visualizer import visualize_cameras
from ggrt.visualization.feature_visualizer import *
from utils_loc import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr, data_shim
from train_ibrnet import synchronize
from ggrt.base.trainer import BaseTrainer
import ggrt.config as config
from ggrt.loss.criterion import MaskedL2ImageLoss, self_sup_depth_loss
from ggrt.loss.photometric_loss import MultiViewPhotometricDecayLoss
from einops import rearrange

import wandb 

from omegaconf import DictConfig
from ggrt.global_cfg import set_cfg
import sys
import configargparse
import hydra
import argparse
from torch.utils.data import DataLoader
import torch
import tqdm
import numpy as np
from dust3r.utils.image import resize_dust, rgb
from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from ggrt.data_loaders import dataset_dict
import os
from ggrt.config import config_parser
from PIL.ImageOps import exif_transpose
# sys.path.append('../')
from dust3r.utils.device import to_numpy
from utils_loc import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr, data_shim
import imageio
from ggrt.data_loaders.geometryutils import relative_transformation
from ggrt.pose_util import Pose, rotation_distance
from finetune_pf_mvsplat_stable import evaluate_camera_alignment
from ggrt.model.dust_gs import dust_gs 

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import matplotlib.pyplot as pl
from scipy.spatial.transform import Rotation
import trimesh
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
import wandb 

from ggrt.geometry.align_poses import align_ate_c2b_use_a2b


class dust2gsTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.state = 'pose_only'
        self.projector = Projector(device=self.device)
        self.photometric_loss = MultiViewPhotometricDecayLoss()
    def build_networks(self):
        self.model = dust_gs(self.config,
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
    def pose_loss(self, pred_pose, gt_pose):
        pred_rt = pred_pose[:,:,:3,:3]
        _,v,_,_ = pred_pose.shape
        gt_rt = gt_pose[:,:,:3,:3]
        pred_rt_T = pred_rt.permute(0, 1,3,2)
        rt_loss_all = 0
        tf_loss_all = 0
        for i in range(v):
            rt_r = torch.matmul(pred_rt_T.squeeze(0)[i], gt_rt.squeeze(0)[i])
            rt_loss = (rt_r.trace()-1)/2
            rt_loss = torch.arccos(rt_loss)
            rt_loss_all = rt_loss_all + rt_loss
            tf_loss = torch.norm(pred_pose.squeeze(0)[i,:3,3] - gt_pose.squeeze(0)[i,:3,3], dim=0)
            tf_loss_all = tf_loss_all + tf_loss
        # Compute the rotation distance between predicted and ground truth poses.
        # Compute the translation distance between predicted and ground truth poses.
        pose_loss = tf_loss_all + rt_loss_all
        return pose_loss
    def compute_geodesic_distance_from_two_matrices(self,m1, m2):
        batch=m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
        
        theta = torch.acos(cos)

        return theta.mean()


    @torch.no_grad()
    def evaluate_camera_alignment(self,aligned_pred_poses, poses_gt):
        # measure errors in rotation and translation
        R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
        R_gt, t_gt = poses_gt.split([3, 1], dim=-1)
        
        R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
        t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)
        
        return R_error, t_error

    def train_iteration(self, batch) -> None:
        self.optimizer.zero_grad()
        batch = data_shim(batch, device=self.device)
        batch = self.model.gaussian_model.data_shim(batch)
        device = self.config.device
        silent=self.config.silent
        model = load_model(self.config.weights, self.config.device, verbose=not self.config.silent)

        batch_size = 1
        with torch.no_grad():

            # rgb_path = batch['rgb_path'][0]
            imgs = resize_dust(batch["context"]["dust_img"],size=512)   #将image resize成512
            pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
            output,feat1,feat2 = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
            mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
            schedule = 'linear'
            lr = 0.01
            if mode == GlobalAlignerMode.PointCloudOptimizer:
                loss = scene.compute_global_alignment(init='mst', niter=0, schedule=schedule, lr=lr)
            # outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                        #   clean_depth, transparent_cams, cam_size)

            # if mode == GlobalAlignerMode.PointCloudOptimizer:
            #     loss = scene.compute_global_alignment(init='mst', niter=0, schedule=schedule, lr=lr)
            rgbimg = scene.imgs
            depths = scene.get_depthmaps()

            poses = scene.get_im_poses()
            poses = poses.detach().cpu()
            # poses_rel=relative_transformation(
            #     poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            #     poses,
            #     orthogonal_rotations=False,
            # )
            poses_rel = poses
            #scene.im_poses 四个pose
            confs = [to_numpy(c) for c in scene.im_conf]    #在每个视角下 点云的置信度
            cmap = pl.get_cmap('jet')
            depths_max = max([d.max() for d in depths])
            depths = [d.unsqueeze(0) for d in depths]
            confs_max = max([d.max() for d in confs])
            confs = [torch.from_numpy(cmap(d/confs_max)).permute(2, 0, 1).unsqueeze(0) for d in confs]
            confs = torch.mean(torch.stack(confs), dim=0)
            depths = torch.cat(depths, dim=0)
            feature_list = []
            depths_list = []
            confs_list = []
            pose_list = []
            intrinsics_list = [] 
            near_list = []
            far_list = []
            context_list = []
            poses_rel = poses_rel.cuda()
            # poses_rel = align_ate_c2b_use_a2b(poses_rel, batch['context']["extrinsics"][0])
            for i in range(len(imgs)-2):            #不添加最后一个target_view  将最后一个ref当成target_view
                features=torch.cat([feat1[i],feat2[i]],dim=0)
                feature_list.append(features.unsqueeze(0))   # b,v,h*w,dim
                confs_list.append(confs[:,i:i+2,:,:])   #b,v,h,w #v=2
                depths_list.append(depths[i:i+2].unsqueeze(0))
                pose_list.append(poses_rel[i:i+2].unsqueeze(0))
                intrinsics_list.append(batch['context']["intrinsics"][:,i:i+2,:,:])
                near_list.append(batch['context']["near"][:,i:i+2])
                far_list.append(batch['context']["far"][:,i:i+2])
                context_list.append(batch['context']["image"][:,i:i+2,:,:,:])
            batch['context']["image"] = torch.cat(context_list,dim=0)
            batch['context']["near"] = torch.cat(near_list,dim=0)
            batch['context']["far"]  = torch.cat(far_list,dim=0)
            batch['context']["intrinsics"] = torch.cat(intrinsics_list,dim=0)
            poses_2_rel = torch.cat(pose_list,dim=0)
            cmap = torch.cat(confs_list,dim=0).cuda()     #两种尝试 一种求平均 第二种等效pixelsplat，在1view下的1点云的置信度，在2view下的2点云的置信度
            features = torch.cat(feature_list,dim=0)
            depths = torch.cat(depths_list,dim=0)
            _,_,_,H,W = batch["context"]['image'].shape
            # features = features.permute(0, 1, 3, 2)
            # features = rearrange(features, "b v d (h w) -> b v d h w",h=H//16,w=W//16)
            batch['target']["extrinsics"] = poses_rel[-2:-1].unsqueeze(0)
            batch['target']["image"] = batch['context']["image"][:,-2:-1,:,:,:]
            # for j,depth in enumerate(depths):
            #     depth = depth.detach()
            #     depth = colorize(depth[0], cmap_name='jet', append_cbar=True)
            #     imageio.imwrite( f'{4*i+j}_dgassin_color_depth.png',
            #                     (depth.cpu().numpy() * 255.).astype(np.uint8))
            pose_error = evaluate_camera_alignment(poses_rel, batch['context']["extrinsics"][0])
            R_errors = pose_error['R_error_mean']
            t_errors = pose_error['t_error_mean']    
        ret, data_gt,_,_ = self.model.gaussian_model(batch,features,poses_2_rel,depths,cmap.float(),self.iteration)
        sfm_loss = 0
        loss_all, loss_dict = 0, {}
        coarse_loss = self.rgb_loss(ret, data_gt)
        # pose_loss = self.pose_loss(ret['ex'], data_gt['ex'])
        loss_dict['gaussian_loss'] = coarse_loss
        loss_all += loss_dict['gaussian_loss']
        loss_all.backward()
        self.optimizer.step()
        self.scheduler.step()
        if self.config.local_rank == 0 and self.iteration % self.config.n_tensorboard == 0:
            mse_error = img2mse(ret['rgb'], data_gt['rgb']).item()
            self.scalars_to_log['train/coarse-loss'] = mse_error
            self.scalars_to_log['train/coarse-psnr'] = mse2psnr(mse_error)
            # self.scalars_to_log['loss/final'] = loss_all.item()
            self.scalars_to_log['loss/rgb_coarse'] = coarse_loss.detach().item()
            print(f"corse loss: {mse_error}, psnr: {mse2psnr(mse_error)},R_errors: {R_errors}, t_errors: {t_errors}")
            self.scalars_to_log['lr/Gaussian'] = self.scheduler.get_last_lr()[0]
            # print(f"R_errors: {R_errors}, t_errors: {t_errors}")
            # aligned_pred_poses, poses_gt = align_predicted_training_poses(
            #     pred_rel_poses[:, -1, :], self.train_data, self.train_dataset, self.config.local_rank)
            # pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
            # visualize_cameras(self.visdom, step=self.iteration, poses=[aligned_pred_poses, poses_gt], cam_depth=0.1)
            self.scalars_to_log['train/R_error_mean'] = pose_error['R_error_mean']
            self.scalars_to_log['train/t_error_mean'] = pose_error['t_error_mean']
            self.scalars_to_log['train/R_error_med'] = pose_error['R_error_med']
            self.scalars_to_log['train/t_error_med'] = pose_error['t_error_med']
            if self.config.expname != 'debug':
                wandb.log(self.scalars_to_log)

    def validate(self) -> float:
        self.model.switch_to_eval()
        torch.cuda.empty_cache()

        target_image = self.train_data['rgb'].squeeze(0).permute(2, 0, 1)
        # pred_inv_depth_gray = self.pred_inv_depth.squeeze(0).detach().cpu()
        # pred_inv_depth = self.pred_inv_depth.squeeze(0).squeeze(0)
        # pred_depth= inv2depth(pred_inv_depth)
        # pred_depth_color = colorize(pred_depth.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)

        self.writer.add_image('train/target_image', target_image, self.iteration)
        # self.writer.add_image('train/pred_inv_depth', pred_inv_depth_gray, self.iteration)
        # self.writer.add_image('train/pred_depth-color', pred_depth_color, self.iteration)

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
def get_predicted_training_poses(pred_poses):
    target_pose = torch.eye(4, device=pred_poses.device, dtype=torch.float).repeat(1, 1, 1)

    # World->camera poses.
    pred_poses = Pose.from_vec(pred_poses) # [n_views, 4, 4]
    pred_poses = torch.cat([target_pose, pred_poses], dim=0)

    # Convert camera poses to camera->world.
    pred_poses = pred_poses.inverse()

    return pred_poses


@torch.no_grad()
def align_predicted_training_poses(pred_poses, data, dataset, device):
    target_pose_gt = data['camera'][..., -16:].reshape(1, 4, 4)
    src_poses_gt = data['src_cameras'][..., -16:].reshape(-1, 4, 4)
    poses_gt = torch.cat([target_pose_gt, src_poses_gt], dim=0).to(device).float()
    
    pred_poses = get_predicted_training_poses(pred_poses)

    aligned_pred_poses = align_ate_c2b_use_a2b(pred_poses, poses_gt)

    return aligned_pred_poses, poses_gt


@torch.no_grad()
def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    # measure errors in rotation and translation
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)
    
    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)
    
    mean_rotation_error = np.rad2deg(R_error.mean().cpu())
    mean_position_error = t_error.mean()
    med_rotation_error = np.rad2deg(R_error.median().cpu())
    med_position_error = t_error.median()
    
    return {'R_error_mean': mean_rotation_error, "t_error_mean": mean_position_error,
            'R_error_med': med_rotation_error, 't_error_med': med_position_error}


@torch.no_grad()
def log_view_to_tb(writer, global_step, args, model, render_stride=1, prefix='', data=None, dataset=None, device=None) -> float:

    pred_inv_depth, pred_rel_poses, _, __ = model.correct_poses(
                            fmaps=None,
                            target_image=data['rgb'].cuda(),
                            ref_imgs=data['src_rgbs'].cuda(),
                            target_camera=data['camera'].cuda(),
                            ref_cameras=data['src_cameras'].cuda(),
                            min_depth=data['depth_range'][0][0],
                            max_depth=data['depth_range'][0][1],
                            scaled_shape=data['scaled_shape'])
    inv_depth_prior = pred_inv_depth.reshape(-1, 1).detach().clone()

    if prefix == 'val/':
        pred_inv_depth = pred_inv_depth.squeeze(0).squeeze(0)
        pred_inv_depth = colorize(pred_inv_depth.detach().cpu(), cmap_name='jet', append_cbar=True).permute(2, 0, 1)
        writer.add_image(prefix + 'pred_inv_depth', pred_inv_depth, global_step)
        aligned_pred_poses, poses_gt = align_predicted_training_poses(pred_rel_poses, data, dataset, args.local_rank)
        pose_error = evaluate_camera_alignment(aligned_pred_poses, poses_gt)
        writer.add_scalar('val/R_error_mean', pose_error['R_error_mean'], global_step)
        writer.add_scalar('val/t_error_mean', pose_error['t_error_mean'], global_step)
        writer.add_scalar('val/R_error_med', pose_error['R_error_med'], global_step)
        writer.add_scalar('val/t_error_med', pose_error['t_error_med'], global_step)

    batch = data_shim(data, device=device)
    batch = model.gaussian_model.data_shim(batch)
    ret, data_gt,ret_ref,data_ref = model.gaussian_model(batch, global_step)
        

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

    if args.local_rank == 0 and args.expname != 'debug':
        wandb.init(
            entity="lifuguan",
            project="mvsplat",
            name=args.expname,
            config=dict(args),
        )
    device = "cuda:{}".format(args.local_rank)

    trainer = dust2gsTrainer(args)
    trainer.train()

if __name__ == '__main__':
    train()
