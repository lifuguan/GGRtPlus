from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict
from .costvolume.ldm_unet.unet import UNetModel
from ....dataset.shims.bounds_shim import apply_bounds_shim
from ....dataset.shims.patch_shim import apply_patch_shim
from ....dataset.types import BatchedExample, DataShim
from ....geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ....global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding
# from .epipolar.conversions import relative_disparity_to_depth


def relative_disparity_to_depth(
    relative_disparity: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert relative disparity, where 0 is near and 1 is far, to depth."""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    return 1 / ((1 - relative_disparity) * (disp_near - disp_far) + disp_far + eps)


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class encoderdust2gs(Encoder[EncoderCostVolumeCfg]):
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)
        # gaussians convertor
        proj_in_channels = 512  #dust feature-dim
        upsample_out_channels = 128

        self.upsampler1 = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.upsampler2 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, 1, 1),
            nn.Upsample(
                scale_factor=8,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.num_channels = 1
        feature_dim = 256
        last_dim = 32 
        self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0)
            )

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        # self.to_gaussians = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(
        #         cfg.d_feature,
        #         cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
        #     ),
        # )
        depth_unet_feat_dim = cfg.depth_unet_feat_dim
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = 32
        depth_unet_channel_mult=cfg.depth_unet_channel_mult
        depth_unet_attn_res=cfg.depth_unet_attn_res,
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
                    num_frames=2,
                    use_cross_view_self_attn=True,
                ),
            )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, 32, 3, 1, 1
        )
        # self.high_resolution_skip = nn.Sequential(
        #     nn.Conv2d(3, cfg.d_feature, 7, 1, 3),
        #     nn.ReLU(),
        # )
        feature_channels = 128
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        gaussian_raw_channels = 84
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        features,
        cnns,
        poses_rel,
        depths,
        densities,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        b, v, _ , _, _ = features.shape

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        features = rearrange(features, "b v d_feature h w -> (b v) d_feature h w ")
        densities = self.head(features)



        cnns = rearrange(cnns, "b v d_feature h w -> (b v) d_feature h w ")
        cnns = self.upsampler2(cnns)
        features = self.upsampler1(torch.cat((features,cnns),dim=1))     #卷积降dim维度  插值h w
        # skip = rearrange(context["image"], "b v c h w -> (b v) c h w")
        # skip = self.high_resolution_skip(skip)
        # features = features + skip
        proj_feature = self.proj_feature(features)

        # depths = rearrange(depths.unsqueeze(-1), "b v h w l -> (b v) l h w")
        context["image"] = rearrange(context["image"], "b v c h w -> (b v) c h w")
        refine_out = self.refine_unet(torch.cat([context["image"],proj_feature,rearrange(depths.unsqueeze(-1), "b v h w l -> (b v) l h w"),densities],dim=1))
        raw_gaussians_in = [refine_out, context["image"],features]
        raw_gaussians_in = torch.cat(raw_gaussians_in,dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(
                raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
            )
        
        features = rearrange(features, "(b v) d_feature h w -> b v (h w) d_feature",b=b,v=v)
        depths = rearrange(depths.unsqueeze(-1).unsqueeze(-1), "b v h w k l -> b v (h w) k l")
        # depths = relative_disparity_to_depth(
        #     rearrange(depths.unsqueeze(-1).unsqueeze(-1), "b v h w k l -> b v (h w) k l"),
        #     rearrange(context["near"], "b v -> b v () () ()"),
        #     rearrange(context["far"], "b v -> b v () () ()"),
        # )
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )

        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        densities = rearrange(densities.unsqueeze(-1), "(b v) l h w k  -> b v (h w) k l", b =b, v=v)
        gaussians = self.gaussian_adapter.forward(
            rearrange(poses_rel, "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            # rearrange(depths),
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )
        
        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
