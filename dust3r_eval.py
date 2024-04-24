import sys
import hydra
from pathlib import Path
from omegaconf import DictConfig

# sys.path.append('./')
# sys.path.append('../')
import visdom
import imageio
import lpips
from eval_dbarf import compose_state_dicts
from einops import rearrange, repeat
from matplotlib import cm
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from ggrt.global_cfg import set_cfg
from ggrt.base.checkpoint_manager import CheckPointManager
from ggrt.model.dbarf import DBARFModel
from utils_loc import *
from ggrt.projection import Projector
from ggrt.data_loaders import dataset_dict
from ggrt.loss.ssim_torch import ssim as ssim_torch
from ggrt.geometry.depth import inv2depth
from ggrt.model.mvsplat.decoder import get_decoder
from ggrt.model.mvsplat.encoder import get_encoder
from ggrt.model.mvsplat.mvsplat import MvSplat

from ggrt.pose_util import Pose
from ggrt.geometry.align_poses import align_ate_c2b_use_a2b
from ggrt.pose_util import rotation_distance
from ggrt.visualization.pose_visualizer import visualize_cameras
from ggrt.model.mvsplat.ply_export import export_ply


mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)

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
def align_predicted_training_poses(pred_poses, data, device='cpu'):
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
    
    return R_error, t_error


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''

    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def img2ssim(gt_image, pred_image):
    """
    Args:
        gt_image: [B, 3, H, W]
        pred_image: [B, 3, H, W]
    """
    return ssim_torch(gt_image, pred_image).item()


def img2lpips(lpips_loss, gt_image, pred_image):
    return lpips_loss(gt_image * 2 - 1, pred_image * 2 - 1).item()


def compose_state_dicts(model) -> dict:
    state_dicts = dict()
    
    state_dicts['net_coarse'] = model.net_coarse
    state_dicts['feature_net'] = model.feature_net
    if model.net_fine is not None:
        state_dicts['net_fine'] = model.net_fine
    state_dicts['pose_learner'] = model.pose_learner

    return state_dicts



def apply_color_map(
    x,
    color_map,
):
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)
    
def apply_color_map_to_image(
    image,
    color_map ,
) :
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")
def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")


    



@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="finetune_mvsplat_stable",
)
def eval(cfg_dict: DictConfig):
    args = cfg_dict
    set_cfg(cfg_dict)
    args.distributed = False


    model = dust_gs(args.config,
                        load_opt=not args.config.no_load_opt,
                        load_scheduler=not args.config.no_load_scheduler,
                        pretrained=args.config.pretrained)


    # Create IBRNet model
    model = DBARFModel(args, load_scheduler=False, load_opt=False, pretrained=False)
    state_dicts = compose_state_dicts(model=model)
    ckpt_manager = CheckPointManager()
    start_step = ckpt_manager.load(config=args, models=state_dicts)
    print(f'start_step: {start_step}')
    
    encoder, encoder_visualizer = get_encoder(args.mvsplat.encoder)
    decoder = get_decoder(args.mvsplat.decoder)
    gaussian_model = MvSplat(encoder, decoder, encoder_visualizer)
    gaussian_model.load_state_dict(torch.load(args.ckpt_path)['gaussian'])
    gaussian_model.cuda()
    
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(args.rootdir, args.expname)
    print("saving results to {}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)


    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, start_step))
    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    results_dict = {scene_name: {}}
    sum_coarse_psnr = 0
    sum_fine_psnr = 0
    running_mean_coarse_psnr = 0
    running_mean_fine_psnr = 0
    sum_coarse_lpips = 0
    sum_fine_lpips = 0
    running_mean_coarse_lpips = 0
    running_mean_fine_lpips = 0
    sum_coarse_ssim = 0
    sum_fine_ssim = 0
    running_mean_coarse_ssim = 0
    running_mean_fine_ssim = 0

    lpips_loss = lpips.LPIPS(net="alex").cuda()
    R_errors = []
    t_errors = []
    video_rgb_pred = []
    video_depth_pred = []
    visdom_ins = visdom.Visdom(server='localhost', port=8097, env='splatam')
    device = "cuda:0"
    projector = Projector(device=device)
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




