import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel
from ggrt.optimizer import BasicUpdateBlockPose, BasicUpdateBlockDepth, DepthHead, PoseHead, UpMaskNet
from ggrt.base.functools import partial
from ggrt.geometry.depth import inv2depth, disp_to_depth
from ggrt.projection import Projector
from ggrt.pose_util import Pose
from ggrt.geometry.camera import Camera
from ggrt.model.feature_network import ResNetEncoder
def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature

def prepare_feat_data_lists(features, intrinsics, near, far, num_samples):
    # prepare features
    b, v, _, h, w = features.shape
    feat_list = []
    feat_ref_lists = []
    init_view_order = list(range(v))
    # feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx+1:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_ref_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)
        feat_list.append(features[0][idx:idx+1])
        # calculate reference pose
        # NOTE: not efficient, but clearer for now
    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_list,feat_ref_lists, intr_curr, depth_candi_curr


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        cur_ref_pose_to_v0_list = []
        for v0, v1 in zip(init_view_order, cur_view_order):
            cur_ref_pose_to_v0_list.append(
                extrinsics[:, v1].clone().detach().inverse()
                @ extrinsics[:, v0].clone().detach()
            )
        cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
        pose_curr_lists.append(cur_ref_pose_to_v0s)

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr


def get_train_poses(query_pose, pred_rel_poses):
    """
    Args:
        target_camera: intrinsics and extrinsics of target view. [1, 34]
        cameras: intrinsics and extrinsics of nearby views. [1, N_views, 34]
        pred_rel_poses: relative poses from target view to nearby views. [N_views, 6]
    """
    num_views = query_pose.shape[0]
    
    R_target, t_target = query_pose[:, :3, :3], query_pose[:, :3, 3].unsqueeze(-1)

    pred_rel_poses = Pose.from_vec(pred_rel_poses) # [n_views, 4, 4]
    R_rel, t_rel = pred_rel_poses[..., :3, :3], pred_rel_poses[..., :3, 3].unsqueeze(-1)

    T_refs = torch.eye(4, dtype=torch.float32, device=pred_rel_poses.device).repeat(num_views, 1, 1)
    R_refs = R_target @ R_rel.permute(0, 2, 1)
    t_refs = (t_target - R_refs @ t_rel)

    T_refs[:, :3, :3], T_refs[:, :3, 3] = R_refs, t_refs.squeeze(-1)

    return T_refs

class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # ablation settings
        # Table 3: base
        self.wo_depth_refine = wo_depth_refine
        # Table 3: w/o cost volume
        self.wo_cost_volume = wo_cost_volume
        # Table 3: w/o U-Net
        self.wo_cost_volume_refine = wo_cost_volume_refine
        self.num_iters = 3
        #添加的iponet
        self.seq_len = 3
        pretrained=True
        self.foutput_dim = 128
        self.feat_ratio = 4
        self.hdim = 128 
        self.cdim = 32
        # self.depth_head = DepthHead(input_dim=self.foutput_dim, hidden_dim=self.foutput_dim, scale=False)
        self.pose_head = PoseHead(input_dim=self.foutput_dim * 2, hidden_dim=self.foutput_dim)
        # self.upmask_net = UpMaskNet(hidden_dim=self.foutput_dim, ratio=self.feat_ratio)
        self.update_block_pose = BasicUpdateBlockPose(hidden_dim=self.hdim, cost_dim=self.foutput_dim, context_dim=self.cdim)
        self.cnet_pose = ResNetEncoder(out_chs=self.hdim+self.cdim, stride=self.feat_ratio, pretrained=pretrained, num_input_images=2)
        # Cost volume refinement: 2D U-Net
        input_channels = feature_channels if wo_cost_volume else (num_depth_candidates + feature_channels)
        channels = self.regressor_feat_dim
        if wo_cost_volume_refine:
            self.corr_project = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            modules = [
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,
                    attention_resolutions=costvolume_unet_attn_res,
                    channel_mult=costvolume_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
                nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
            ]
            self.corr_refine_net = nn.Sequential(*modules)
            # cost volume u-net skip connection
            self.regressor_residual = nn.Conv2d(
                input_channels, num_depth_candidates, 1, 1, 0
            )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # CNN-based feature upsampler
        proj_in_channels = feature_channels + feature_channels
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        if wo_depth_refine:  # for ablations
            self.refine_unet = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            self.refine_unet = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=depth_unet_attn_res,
                    channel_mult=depth_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
            )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        if not wo_depth_refine:
            channels = depth_unet_feat_dim
            disps_models = [
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
            ]
            self.to_disparity = nn.Sequential(*disps_models)
    def upsample_depth(self, depth, mask, ratio=8, image_size=None):
        """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = depth.shape
        mask = mask.view(N, 1, 9, ratio, ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(depth, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        # return up_flow.reshape(N, 1, ratio*H, ratio*W)

        up_flow = up_flow.reshape(N, 1, ratio*H, ratio*W)
        up_flow = F.interpolate(up_flow, size=image_size, mode='bilinear', align_corners=True)
    
        return up_flow
    def get_cost_each(self, pose, fmap, fmap_ref, depth, K, ref_K, scale_factor):
        """
            depth: (b, 1, h, w)
            fmap, fmap_ref: (b, c, h, w)
        """
        pose = Pose.from_vec(pose)

        device = depth.device
        cam = Camera(K=K.float()).scaled(scale_factor).to(device) # tcw = Identity
        ref_cam = Camera(K=ref_K.float(), Twc=pose).scaled(scale_factor).to(device)
        
        # Reconstruct world points from target_camera
        world_points = cam.reconstruct(depth, frame='w')
        
        # Project world points onto reference camera
        ref_coords = ref_cam.project(world_points, frame='w', normalize=True) #(b, h, w,2)

        fmap_warped = F.grid_sample(fmap_ref, ref_coords, mode='bilinear', padding_mode='zeros', align_corners=True) # (b, c, h, w)
        
        cost = (fmap - fmap_warped)**2
    
        return cost

    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
        image_size=None,
        context_iamge=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        # format the input
        b, v, c, h, w = features.shape
        fmaps_ref_list =[]
        fmap1, fmaps_ref = features[0][0:1], features[0][1:]   #参考帧第一帧为fmap1，其余为fmaps_ref # 1 128 124 188 // 3 128 124 188
        # self.scale_inv_depth = partial(disp_to_depth, min_depth=near[0][0], max_depth=far[0][0])
        device = features.device
        depth_list = []
        for iter in range(self.num_iters):
            if iter == 0:
                if cnn_features is not None:
                    cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")
                target_image = context_iamge[0][0:1]
                ref_imgs = context_iamge[0][1:]
                for i in range(v-1):
                    fmaps_ref_list.append(fmaps_ref[i:i+1])
                pose_predictions = []
                for fmap_ref in fmaps_ref_list:
                    pose_predictions.append(self.pose_head(torch.cat([fmap1, fmap_ref], dim=1)))
                img_pairs = []
                for i in range(v - 1):
                    ref_img = ref_imgs[i].unsqueeze(0)
                    img_pairs.append(torch.cat([target_image, ref_img], dim=1))
                cnet_pose_list = self.cnet_pose(img_pairs)   #这里使用原分辨率试试
                hidden_p_list, inp_p_list = [], []
                for cnet_pose in cnet_pose_list:
                    hidden_p, inp_p = torch.split(cnet_pose, [self.hdim, self.cdim], dim=1)
                    hidden_p_list.append(torch.tanh(hidden_p))
                    inp_p_list.append(torch.relu(inp_p))
                pose_predictions_list = []
                pose_predictions_list.append(torch.cat(pose_predictions,dim=0).unsqueeze(0))
            pose_list = pose_predictions
            if iter > 0 :
                pose_list = [pose.detach() for pose in pose_list]
                pose_cost_func_list = []
                feat_size = features.shape[-2:]
                depth_project = depth_project.detach()
                depth_project = F.interpolate(depth_project, size=feat_size, mode='bilinear', align_corners=True)  # 下采样
                for i, fmap_ref in enumerate(fmaps_ref_list):
                    pose_cost_func_list.append(partial(self.get_cost_each, fmap=fmap1, fmap_ref=fmap_ref,
                                                    depth=depth_project,  #利用第一帧的depth
                                                    K=intrinsics[0][0:1], ref_K=intrinsics[0][i+1], scale_factor=1.0/self.feat_ratio))


                    #########  update pose ###########
                pose_list_seqs = [None] * len(pose_list)
                pose_predictions = []
                for i, (pose, hidden_p) in enumerate(zip(pose_list, hidden_p_list)):
                    hidden_p, pose_seqs = self.update_block_pose(hidden_p, pose_cost_func_list[i],
                                                                pose, inp_p_list[i], seq_len=self.seq_len)
                    hidden_p_list[i] = hidden_p

                    pose_seqs = pose_seqs[-1]
                    pose_predictions.append(pose_seqs) 
                pose_predictions_list.append(torch.cat(pose_predictions,dim=0).unsqueeze(0))
            num_views = v-1
            query_pose = torch.eye(4).unsqueeze(0).repeat(num_views, 1, 1).to(device)  # [n_views, 4, 4]
            extrinsics_pred=get_train_poses(query_pose,torch.cat(pose_predictions,dim=0))
            extrinsics_pred= torch.cat([query_pose[0:1],extrinsics_pred],dim=0)
            extrinsics_pred = extrinsics_pred.unsqueeze(0)
            if True:
                feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
                    prepare_feat_proj_data_lists(
                        features,
                        intrinsics,
                        extrinsics_pred, # ext_pred
                        near,
                        far,
                        num_samples=self.num_depth_candidates,
                    )
                )

            

        # cost volume constructions
            feat01 = feat_comb_lists[0]
            if self.wo_cost_volume:
                raw_correlation_in = feat01
            else:
                raw_correlation_in_lists = []
                for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                    # sample feat01 from feat10 via camera projection
                    feat01_warped = warp_with_pose_depth_candidates(                    
                        feat10,
                        intr_curr,
                        pose_curr,
                        1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                        warp_padding_mode="zeros",
                    )  # [B, C, D, H, W]
                    # calculate similarity
                    raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                        1
                    ) / (
                        c**0.5
                    )  # [vB, D, H, W]
                    raw_correlation_in_lists.append(raw_correlation_in)
                # average all cost volumes
                raw_correlation_in = torch.mean(
                    torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
                )  # [vxb d, h, w]
                raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)

            # refine cost volume via 2D u-net
            if self.wo_cost_volume_refine:
                raw_correlation = self.corr_project(raw_correlation_in)
            else:
                raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
                # apply skip connection
                raw_correlation = raw_correlation + self.regressor_residual(
                    raw_correlation_in
                )

            # softmax to get coarse depth and density
            pdf = F.softmax(
                self.depth_head_lowres(raw_correlation), dim=1
            )  # [2xB, D, H, W]
            coarse_disps = (disp_candi_curr * pdf).sum(
                dim=1, keepdim=True
            )  # (vb, 1, h, w)
            pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
            pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
            fullres_disps = F.interpolate(
                coarse_disps,
                scale_factor=self.upscale_factor,
                mode="bilinear",
                align_corners=True,
            )

            #得到depth后计算 pose cost

            # depth refinement
            proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
            proj_feature = self.proj_feature(proj_feat_in_fullres)
            refine_out = self.refine_unet(torch.cat(
                (extra_info["images"], proj_feature, fullres_disps, pdf_max), dim=1
            ))

            # gaussians head
            raw_gaussians_in = [refine_out,
                                extra_info["images"], proj_feat_in_fullres]
            raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
            raw_gaussians = self.to_gaussians(raw_gaussians_in)
            raw_gaussians = rearrange(
                raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
            )

            if self.wo_depth_refine:
                densities = repeat(
                    pdf_max,
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )
                depths = 1.0 / fullres_disps
                depth_project = depths[0:1]
                depths = repeat(
                    depths,
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )
            else:
                # delta fine depth and density
                delta_disps_density = self.to_disparity(refine_out)
                delta_disps, raw_densities = delta_disps_density.split(
                    gaussians_per_pixel, dim=1
                )

                # combine coarse and fine info and match shape
                densities = repeat(
                    F.sigmoid(raw_densities),
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )
                fine_disps = (fullres_disps + delta_disps).clamp(
                    1.0 / rearrange(far, "b v -> (v b) () () ()"),
                    1.0 / rearrange(near, "b v -> (v b) () () ()"),
                )
                depths = 1.0 / fine_disps
                depth_project = depths[0:1]
                depth_list.append(depth_project)
                depths = repeat(
                    depths,
                    "(v b) dpt h w -> b v (h w) srf dpt",
                    b=b,
                    v=v,
                    srf=1,
                )

        return depths, densities, raw_gaussians,extrinsics_pred,torch.cat(pose_predictions_list,dim=0).permute(1,0,2) ,depth_list
