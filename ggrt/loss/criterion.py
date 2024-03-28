# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

from utils_loc import img2mse
from ggrt.geometry.depth import depth2inv


class MaskedL2ImageLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        if 'mask' in outputs:
            pred_mask = outputs['mask'].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss


def pseudo_huber_loss(residual, scale=10):
    trunc_residual = residual / scale
    return torch.sqrt(trunc_residual * trunc_residual + 1) - 1


class FeatureMetricLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, target_rgb_feat, nearby_view_rgb_feat, mask=None):
        '''
        Args:
            target_rgb_feat: [n_rays, n_samples=1, n_views+1, d+3]
            nearby_view_rgb_feat: [n_rays, n_samples=1, n_views+1, d+3]
        '''
        if mask is None:
            l1_loss = nn.L1Loss(reduction='mean')
            # mse_loss = nn.MSELoss(reduction='mean')
            # loss = mse_loss(nearby_view_rgb_feat, target_rgb_feat)

            loss = l1_loss(nearby_view_rgb_feat, target_rgb_feat)
        
        else:
            feat_diff = target_rgb_feat - nearby_view_rgb_feat
            feat_diff_square = (feat_diff * feat_diff).squeeze(1)
            mask = mask.repeat(1, 1, 1).permute(2, 0, 1)
            n_views, n_dims = target_rgb_feat.shape[-2], target_rgb_feat.shape[-1]
            loss = torch.sum(feat_diff_square * mask) / (torch.sum(mask.squeeze(-1)) * n_views * n_dims + 1e-6)

            # feat_diff_huber = pseudo_huber_loss(feat_diff, scale=0.8).squeeze(1)
            # mask = mask.repeat(1, 1, 1).permute(2, 0, 1)
            # n_views, n_dims = target_rgb_feat.shape[-2], target_rgb_feat.shape[-1]
            # loss = torch.sum(feat_diff_huber * mask) / (torch.sum(mask.squeeze(-1)) * n_views * n_dims + 1e-6)
        
        return loss


def self_sup_depth_loss(inv_depth_prior, rendered_depth, min_depth, max_depth):
    min_disparity = 1.0 / max_depth
    max_disparity = 1.0 / min_depth
    valid = ((inv_depth_prior > min_disparity) & (inv_depth_prior < max_disparity)).detach()

    inv_rendered_depth = depth2inv(rendered_depth)

    loss_depth = torch.mean(valid * torch.abs(inv_depth_prior - inv_rendered_depth))

    return loss_depth


def sup_depth_loss(ego_motion_inv_depths, gt_depth, min_depth, max_depth):
    num_iters = len(ego_motion_inv_depths)
    total_loss = 0
    total_w = 0
    gamma = 0.85
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth

    gt_inv_depth = depth2inv(gt_depth)

    valid = ((gt_inv_depth > min_disp) & (gt_inv_depth < max_disp)).detach()

    for i, inv_depth in enumerate(ego_motion_inv_depths):
        w = gamma ** (num_iters - i - 1)
        total_w += w

        loss_depth = torch.mean(valid * torch.abs(gt_inv_depth - inv_depth.squeeze(0)))
        loss_i = loss_depth
        total_loss += w * loss_i
    loss = total_loss / total_w
    return loss

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def shuffle(input):
    # shuffle dim=1
    idx = torch.randperm(input[0].shape[1])
    for i in range(input.shape[0]):
        input[i] = input[i][:, idx].view(input[i].shape)

def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def loss_depth_grad(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = img_grad_x / (torch.abs(img_grad_x) + 1e-6)
    weight_y = img_grad_y / (torch.abs(img_grad_y) + 1e-6)

    depth_grad_x = depth[:, :, :, :-1] - depth[:, :, :, 1:]
    depth_grad_y = depth[:, :, :-1, :] - depth[:, :, 1:, :]
    grad_x = depth_grad_x / (torch.abs(depth_grad_x) + 1e-6)
    grad_y = depth_grad_y / (torch.abs(depth_grad_y) + 1e-6)

    loss = l1_loss(grad_x, weight_x) + l1_loss(grad_y, weight_y)
    return loss


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def margin_l2_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask
    
def margin_l1_loss(network_output, gt, margin, return_mask=False):
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask].abs()).mean()
    else:
        return ((network_output - gt)[mask].abs()).mean(), mask
    

def kl_loss(input, target):
    input = F.log_softmax(input, dim=-1)
    target = F.softmax(target, dim=-1)
    return F.kl_div(input, target, reduction="batchmean")

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_l1_loss_global(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size), std = input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std = target.std().detach())
    return margin_l1_loss(input_patches, target_patches, margin, return_mask)

def patch_norm_l1_loss(input, target, patch_size, margin, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l1_loss(input_patches, target_patches, margin, return_mask)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def margin_ssim(img1, img2, window_size=11, size_average=True):
    result = ssim(img1, img2, window_size, False)
    print(result.shape)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

