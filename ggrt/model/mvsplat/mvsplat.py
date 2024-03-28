from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn, optim

from ...global_cfg import get_cfg
from .types import Gaussians
from ...dataset.data_module import get_data_shim
from ...dataset.types import BatchedExample
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
import numpy as np
from .wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from .interpolatation import interpolate_extrinsics,interpolate_intrinsics

from .ply_export import export_ply

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class MvSplat(nn.Module):
    encoder: nn.Module
    decoder: Decoder

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        encoder_visualizer: Optional[EncoderVisualizer],
    ) -> None:
        super().__init__()
         # Set up the model.
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_visualizer = encoder_visualizer
        
        self.data_shim = get_data_shim(self.encoder)
        self.last_ref_gaussians = {}
            
        self.test_iteration = 0

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.gaussian_model.parameters(), lr=self.config.optimizer.lr)
        warm_up_steps = self.config.optimizer.warm_up_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                        1 / warm_up_steps,
                                                        1,
                                                        total_iters=warm_up_steps)

    def batch_cut(self, batch, idx1, idx2):
        return {
            'extrinsics': torch.cat([batch['extrinsics'][:,idx1:idx1+1,:,:], batch['extrinsics'][:,idx2:idx2+1,:,:]], dim=1),
            'intrinsics': torch.cat([batch['intrinsics'][:,idx1:idx1+1,:,:], batch['intrinsics'][:,idx2:idx2+1,:,:]], dim=1),
            'image': torch.cat([batch['image'][:,idx1:idx1+1,...], batch['image'][:,idx2:idx2+1,...]], dim=1),
            'near': torch.cat([batch['near'][:,idx1:idx1+1], batch['near'][:,idx2:idx2+1]], dim=1),
            'far': torch.cat([batch['far'][:,idx1:idx1+1], batch['far'][:,idx2:idx2+1]], dim=1),
            'index': torch.cat([batch['index'][:,idx1:idx1+1], batch['index'][:,idx2:idx2+1]], dim=1),
        }
    def trajectory_fn(self,batch,t):
            _, v, _, _ = batch["context"]["extrinsics"].shape
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
            # delta = (origin_a - origin_b).norm(dim=-1)
            # tf = generate_wobble_transformation(
            #     delta * 0.5,
            #     t,
            #     5,
            #     scale_radius_with_t=False,
            # )
            index_sort = np.argsort([int(s.item()) for s in batch["context"]["index"][0]])
            start = index_sort[1]
            for i in range(4):
                if  0== batch["context"]["index"][0][i]:
                    start = i
                if  4 == batch["context"]["index"][0][i]:
                    end = i
            ex_end = torch.tensor([[-2.1482e-02,  1.3204e-02,  9.9968e-01,  1.1767e+01],
            [-9.9977e-01,  6.8249e-04, -2.1493e-02, -1.5805e-01],
            [-9.6607e-04, -9.9991e-01,  1.3187e-02,  2.1376e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device='cuda:0')
            extrinsics = interpolate_extrinsics(
                batch["target"]["extrinsics"][0, 0],
                ex_end,
                # batch["context"]["extrinsics"][0, end],
                # if v == 2
                # else batch["target"]["extrinsics"][0, 0],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, start],
                batch["context"]["intrinsics"][0, end],
                # if v == 2
                # else batch["target"]["intrinsics"][0, 0],
                t ,
            )
            return extrinsics[None] , intrinsics[None]
    def trajectory_fn_woble(self,batch,t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics


    def forward(self, batch, global_step):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # index_sort = np.argsort([int(s.item()) for s in batch["context"]["index"][0]])
        # str_current_idx = [str(item) for item in batch["context"]["index"][0].cpu().numpy()]
        # unused_indexs = set(list(self.last_ref_gaussians.keys())) - set(str_current_idx) 
        # if len(unused_indexs) > 0:
        #         for unused_idx in tuple(unused_indexs):
        #            del self.last_ref_gaussians[unused_idx]
        # gaussians = None
        # for k in range(len(index_sort)-1):
        #     if str_current_idx[index_sort[k]] in self.last_ref_gaussians.keys(): # 如果已经计算过，则直接使用
        #             tmp_gaussians = self.last_ref_gaussians[str_current_idx[index_sort[k]]].detach()
        #     else:
        #         tmp_batch = self.batch_cut(batch["context"], index_sort[k], index_sort[k+1])   
        #         tmp_gaussians = self.encoder(
        #             tmp_batch, global_step, False, scene_names=batch["scene"]
        #         )
        #     if gaussians is None:
        #         gaussians: Gaussians = tmp_gaussians
        #     else:
        #         gaussians.covariances = torch.cat([gaussians.covariances, tmp_gaussians.covariances], dim=1)
        #         gaussians.means = torch.cat([gaussians.means, tmp_gaussians.means], dim=1)
        #         gaussians.harmonics = torch.cat([gaussians.harmonics, tmp_gaussians.harmonics], dim=1)
        #         gaussians.opacities = torch.cat([gaussians.opacities, tmp_gaussians.opacities], dim=1)
        gaussians = self.encoder(
            batch["context"], global_step, False, scene_names=batch["scene"]
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode='depth',
        )
        # output_ref = self.decoder.forward(
        #     gaussians,
        #     batch["context"]["extrinsics"],
        #     batch["context"]["intrinsics"],
        #     batch["context"]["near"],
        #     batch["context"]["far"],
        #     (h, w),
        #     depth_mode='depth',
        # )
        ret = {'rgb': output.color, 'depth': output.depth}
        target_gt = {'rgb': batch["target"]["image"]}
        # ret = {'rgb': torch.cat([output.color,output_ref.color],dim=1), 'depth': output.depth}
        # target_gt = {'rgb': torch.cat([batch["target"]["image"],batch["context"]["image"]],dim=1)}
        return ret, target_gt