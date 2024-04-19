from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

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


class encoderdust2gs(Encoder[EncoderCostVolumeCfg]):
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)
        # gaussians convertor
        proj_in_channels = 256   #dust feature-dim
        upsample_out_channels = 128

        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)
        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                cfg.d_feature,
                cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
            ),
        )
        self.high_resolution_skip = nn.Sequential(
            nn.Conv2d(3, cfg.d_feature, 7, 1, 3),
            nn.ReLU(),
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
        features = self.upsampler(features)     #卷积降dim维度  插值h w
        skip = rearrange(context["image"], "b v c h w -> (b v) c h w")
        skip = self.high_resolution_skip(skip)
        features = features + skip

        features = rearrange(features, "(b v) d_feature h w -> b v (h w) d_feature",b=b,v=v)
        gaussians = rearrange(
            self.to_gaussians(features),
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        depths = rearrange(depths.unsqueeze(-1).unsqueeze(-1), "b v h w k l -> b v (h w) k l")
        # depths = relative_disparity_to_depth(
        #     rearrange(depths.unsqueeze(-1).unsqueeze(-1), "b v h w k l -> b v (h w) k l"),
        #     rearrange(context["near"], "b v -> b v () () ()"),
        #     rearrange(context["far"], "b v -> b v () () ()"),
        # )


        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel
        densities = rearrange(densities.unsqueeze(-1).unsqueeze(-1), "b v h w k l -> b v (h w) k l")
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
