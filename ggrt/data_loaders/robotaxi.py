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

import sys
sys.path.append('../')
import glob
import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from .data_utils import get_nearby_view_ids, loader_resize, random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses

class RoboTaxiDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = os.path.join('data/', 'robotaxi/')
        self.dataset_name = 'llff'
        self.pose_noise_level = 0

        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_extrinsic_files = []
        self.train_rgb_files = []
        self.idx_to_node_id_list = []
        self.node_id_to_idx_list = []
        self.train_view_graphs = []
    

        
        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        print(f'[INFO] num scenes: {len(scenes)}')

        self.ORIGINAL_SIZE = (678, 1224)
        # self.image_size = (672, 1216)
        self.image_size = (240, 420)

        fx, fy = 1.139056561710934830e+03, 9.286030859337716947e+02
        cx, cy = 5.306224397627581766e+02, 4.880767184452269021e+02
        fx, fy = (
                fx * self.image_size[1] / self.ORIGINAL_SIZE[1],
                fy * self.image_size[0] / self.ORIGINAL_SIZE[0],
            )
        cx, cy = (
                cx * self.image_size[1] / self.ORIGINAL_SIZE[1],
                cy * self.image_size[0] / self.ORIGINAL_SIZE[0],
            )
        intrinsic = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            
            rgb_files = sorted(glob.glob(os.path.join(scene_path, '*.jpg')))
            intrinsics = np.tile(intrinsic, (len(rgb_files), 1, 1))
            extrinsic_files = [f.replace(".jpg", "_ext.txt") for f in rgb_files]
            near_depth, far_depth = 0.1, 1000.
            
            i_test = np.arange(len(rgb_files))[::self.args.llffhold] if mode != 'eval_pose' else []
            i_train = np.array([j for j in np.arange(len(rgb_files)) if (j not in i_test and j not in i_test)])

            if mode == 'train' or mode == 'eval_pose':
                i_render = i_train
            else:
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_extrinsic_files.append(np.array(extrinsic_files)[i_train].tolist())
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in np.array(extrinsic_files)[i_render].tolist()])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)

    def num_poses(self):
        return len(self.render_rgb_files)

    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode == 'train' else len(self.render_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = np.genfromtxt(self.render_poses[idx], delimiter=',', filling_values=np.nan)[:-1].reshape(4, 4)
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = np.stack([np.genfromtxt(file, delimiter=',', filling_values=np.nan)[:-1].reshape(4, 4) for file in self.train_extrinsic_files[train_set_id]], axis=0)
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views 
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = None
        # num_select = min(self.num_source_views*subsample_factor, 28)
        if self.args.selection_rule == 'pose' or self.mode != 'train':
            nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                    train_poses,
                                                    num_select=num_select,
                                                    tar_id=id_render,
                                                    angular_dist_method='dist')
        elif self.args.selection_rule == 'view_graph':
            nearest_pose_ids = get_nearby_view_ids(target_id=id_render,
                                                   graph=view_graph['graph'],
                                                   idx_to_node_id=idx_to_node_id,
                                                   node_id_to_idx=node_id_to_idx,
                                                   num_select=num_select)
        else:
            raise NotImplementedError
        
        if self.mode == 'eval_pose' and self.nearby_view_id is not None:
            nearest_pose_ids = np.array([self.nearby_view_id])

        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)
        # print(f'nearest pose ids: {nearest_pose_ids}')

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        # relative_poses = None if self.args.selection_rule == 'pose' else \
        #                  get_relative_poses(idx, view_graph['two_view_geometries'], idx_to_node_id, nearest_pose_ids)

        src_rgbs = []
        src_cameras = []
        src_intrinsics, src_extrinsics = [], []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            
            src_intrinsics.append(train_intrinsics_)
            src_extrinsics.append(train_pose)
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        src_intrinsics, src_extrinsics = np.stack(src_intrinsics, axis=0), np.stack(src_extrinsics, axis=0)
    
        pix_rgb, pix_camera, pix_src_rgbs, pix_src_cameras, pix_intrinsics, pix_src_intrinsics = loader_resize(rgb,camera,src_rgbs,src_cameras, size=self.image_size, int_resize=False)
        
       
        pix_src_extrinsics = torch.from_numpy(src_extrinsics).float()
        pix_extrinsics = torch.from_numpy(render_pose).unsqueeze(0).float()
        
        pix_src_intrinsics = self.normalize_intrinsics(torch.from_numpy(pix_src_intrinsics[:,:3,:3]).float(), self.image_size)
        pix_intrinsics = self.normalize_intrinsics(torch.from_numpy(pix_intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size)

        # # intrinsics[:, :2, :2] *= self.ratio
        # src_intrinsics[:, :2, :2]=src_intrinsics[:,:2,:2]*self.ratio
        
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5], dtype=torch.float32)

        # Resize the world to make the baseline 1.
        if pix_src_extrinsics.shape[0] == 2:
            a, b = pix_src_extrinsics[:, :3, 3]
            scale = (a - b).norm()
            if scale < 0.001:
                print(
                    f"Skipped {scene} because of insufficient baseline "
                    f"{scale:.6f}"
                )
            pix_src_extrinsics[:, :3, 3] /= scale
            pix_extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        
        return {'rgb': torch.from_numpy(pix_rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(pix_src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                'idx': idx,
                'scaled_shape': (0, 0), # (378, 504)
                "context": {
                        "extrinsics": pix_src_extrinsics,
                        "intrinsics": pix_src_intrinsics,
                        "image": torch.from_numpy(pix_src_rgbs[..., :3]).permute(0, 3, 1, 2),
                        "near":  depth_range[0].repeat(num_select) / scale,
                        "far": depth_range[1].repeat(num_select) / scale,
                        "index": torch.from_numpy(nearest_pose_ids),
                },
                "target": {
                        "extrinsics": pix_extrinsics,
                        "intrinsics": pix_intrinsics,
                        "image": torch.from_numpy(pix_rgb[..., :3]).unsqueeze(0).permute(0, 3, 1, 2),
                        "near": depth_range[0].unsqueeze(0) / scale,
                        "far": depth_range[1].unsqueeze(0) / scale,
                        "index": id_render,
                },
                }
    def normalize_intrinsics(self, intrinsics, img_size):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized
    def normalize_intrinsics_1(self, intrinsics, img_size,center_h,center_w):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone().float()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] /= w
        intrinsics_normalized[:, 1, 2] /= h
        return intrinsics_normalized