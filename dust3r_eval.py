
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
def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)



@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="finetune_mvsplat_stable",
)

def eval(cfg_dict: DictConfig):
    # args = cfg_dict
    # set_cfg(cfg_dict)
    # args.distributed = False
    # test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)

    args = cfg_dict
    set_cfg(cfg_dict)
    args.distributed = False
    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    extra_out_dir = '{}/{}'.format(args.rootdir, args.expname)
    device = args.device
    silent=args.silent
    model = load_model(args.weights, args.device, verbose=not args.silent)
    batch_size = 1
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{}'.format(scene_name, 'dust_depth'))
    os.makedirs(out_scene_dir, exist_ok=True)
    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        imgs = resize_dust(data["context"]["dust_img"],size=512)   #将image resize成512
        pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)
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
        #scene.im_poses 四个pose
        confs = to_numpy([c for c in scene.im_conf])
        cmap = pl.get_cmap('jet')
        depths_max = max([d.max() for d in depths])
        # depths = [d/depths_max for d in depths]
        confs_max = max([d.max() for d in confs])
        confs = [cmap(d/confs_max) for d in confs]
        for j,depth in enumerate(depths):
            depth = depth.detach()
            depth = colorize(depth, cmap_name='jet', append_cbar=True)
            imageio.imwrite(os.path.join(out_scene_dir, f'{4*i+j}_dgassin_color_depth.png'),
                            (depth.cpu().numpy() * 255.).astype(np.uint8))
        poses = scene.get_im_poses()
        poses = poses.detach().cpu()
        poses_rel=relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )
        pose_error = evaluate_camera_alignment(poses_rel, data['context']["extrinsics"][0])
        R_errors = pose_error['R_error_mean']
        t_errors = pose_error['t_error_mean']
        print(f'R_errors: {R_errors}')
        print(f't_errors: {t_errors}')

if __name__ == '__main__':
    eval()




