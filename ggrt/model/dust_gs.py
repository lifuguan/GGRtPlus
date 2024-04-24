import os
import math

import torch

from ggrt.model.feature_network import ResUNet
from ggrt.depth_pose_network import DepthPoseNet
from ggrt.loss.photometric_loss import MultiViewPhotometricDecayLoss
import tqdm
from ggrt.base.model_base import Model
from dust3r.utils.device import to_cpu, collate_with_cat
from ggrt.model.mvsplat.decoder import get_decoder
from ggrt.model.mvsplat.encoder import get_encoder
from ggrt.model.mvsplat.dustsplat import dustSplat
from dust3r.utils.image import resize_dust, rgb
# from dust3r.inference import infer, load_model
from dust3r.image_pairs import make_pairs
from dust3r.model import AsymmetricCroCo3DStereo, inf 
from dust3r.losses import *
class dust_gs(Model):
    def __init__(self, args, load_opt=True, load_scheduler=True, pretrained=True):
        device = torch.device(f'cuda:{args.local_rank}')
        # create generalized 3d gaussian.

        self.pose_learner = eval(args.model).to(device)
        print(f'>> Creating train criterion = {args.train_criterion}')
        # train_criterion = eval(args.train_criterion)
        # print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
        # test_criterion = eval(args.test_criterion or args.criterion).to(device)
        self.pose_learner.to(device)
        self.args = args


        encoder, encoder_visualizer = get_encoder(args.mvsplat.encoder)
        decoder = get_decoder(args.mvsplat.decoder)
        self.gaussian_model = dustSplat(encoder, decoder, encoder_visualizer)
        self.gaussian_model.to(device)
        self.photometric_loss = MultiViewPhotometricDecayLoss()

    def to_distributed(self):
        super().to_distributed()

        if self.args.distributed:
            self.pose_learner = torch.nn.parallel.DistributedDataParallel(
                self.pose_learner,
                device_ids=[self.args.local_rank],
                output_device=[self.args.local_rank]
            )
            self.gaussian_model = torch.nn.parallel.DistributedDataParallel(
                self.gaussian_model,
                device_ids=[self.args.local_rank],
                output_device=[self.args.local_rank]
            )

    def switch_to_eval(self):
        self.pose_learner.eval()
        self.gaussian_model.eval()

    def switch_to_train(self):
        self.pose_learner.train(True)
        self.gaussian_model.train()
            
    def check_if_same_size(self ,pairs):
        shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
        shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
        return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


    def _interleave_imgs(self,img1, img2):
        res = {}
        for key, value1 in img1.items():
            value2 = img2[key]
            if isinstance(value1, torch.Tensor):
                value = torch.stack((value1, value2), dim=1).flatten(0, 1)
            else:
                value = [x for pair in zip(value1, value2) for x in pair]
            res[key] = value
        return res

    def make_batch_symmetric(self , batch):
        view1, view2 = batch
        view1, view2 = (self._interleave_imgs(view1, view2), self._interleave_imgs(view2, view1))
        return view1, view2

    def loss_of_one_batch(self,batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None):
        view1, view2 = batch
        for view in batch:
            for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        if symmetrize_batch:
            view1, view2 = self.make_batch_symmetric(batch)

        with torch.cuda.amp.autocast(enabled=bool(use_amp)):
            pred1, pred2 ,feat1, feat2 , path_1 ,path_2= model(view1, view2)

            # loss is supposed to be symmetric
            with torch.cuda.amp.autocast(enabled=False):
                # loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None
                loss=0
        result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
        return  result,feat1,feat2,path_1 ,path_2

    def correct_poses(self, batch,device,batch_size,silent):
        """
        Args:
            fmaps: [n_views+1, c, h, w]
            target_image: [1, h, w, 3]
            ref_imgs: [1, n_views, h, w, 3]
            target_camera: [1, 34]
            ref_cameras: [1, n_views, 34]
        Return:
            inv_depths: n_iters*[1, 1, h, w] if training else [1, 1, h, w]
            rel_poses: [n_views, n_iters, 6] if training else [n_views, 6]
        """
        imgs = resize_dust(batch["context"]["dust_img"],size=512)   #将image resize成512
        pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)

        verbose=True


        # if verbose:
        #     print(f'>> Inference with model on {len(pairs)} image pairs')
        result = []

    # first, check if all images have the same size
        multiple_shapes = not (self.check_if_same_size(pairs))
        if multiple_shapes:  # force bs=1
            batch_size = 1
    # batch_size = len(pairs)
        batch_size = 1
        feat1_list = []
        feat2_list = []
        cnn1_list = []
        cnn2_list = []
        for i in range(0, len(pairs), batch_size):
            view1_ft_lst = []
            view2_ft_lst = []
            # start = time.time()
            train_criterion = eval(self.args.train_criterion).to(device)
            loss_tuple,cnn1 ,cnn2,path_1 ,path_2 =  self.loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), self.pose_learner, train_criterion, device,
                                        symmetrize_batch=False,
                                        use_amp=False, ret='loss')
            # res ,cnn1 ,cnn2,path_1 ,path_2= loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), model, None, device)
            # end =time.time()
            # print(end-start)
            result.append(to_cpu(loss_tuple))
            feat1 = path_1
            feat2 = path_2
            feat1_list.append(feat1)
            feat2_list.append(feat2)
            cnn1_list.append(cnn1)
            cnn2_list.append(cnn2)
        # pfeat01.append(dec2[0])
        result = collate_with_cat(result, lists=multiple_shapes)

        return result,feat1_list,feat2_list,cnn1_list,cnn2_list,imgs



        # output,feat1,feat2,cnn1,cnn2 = infer(self.args,pairs, self.pose_learner, device, batch_size=batch_size, verbose=not silent)

        return result,feat1,feat2,cnn1,cnn2,imgs

    def switch_state_machine(self, state='joint') -> str:
        if state == 'pose_only':
            self._set_pose_learner_state(opt=True)
            self._set_gaussian_state(opt=False)
        
        elif state == 'nerf_only':
            self._set_pose_learner_state(opt=False)
            self._set_gaussian_state(opt=True)
        
        elif state == 'joint':
            self._set_pose_learner_state(opt=True)
            self._set_gaussian_state(opt=True)
        
        else:
            raise NotImplementedError("Not supported state")
        
        return state

    def _set_pose_learner_state(self, opt=True):
        for param in self.pose_learner.parameters():
            param.requires_grad = opt

    def _set_gaussian_state(self, opt=True):
        for param in self.gaussian_model.parameters():
            param.requires_grad = opt
    
    def compose_joint_loss(self, sfm_loss, nerf_loss, step, coefficient=1e-5):
        # The jointly training loss is composed by the convex_combination:
        #   L = a * L1 + (1-a) * L2
        alpha = math.pow(2.0, -coefficient * step)
        loss = alpha * sfm_loss + (1 - alpha) * nerf_loss
        
        return loss
