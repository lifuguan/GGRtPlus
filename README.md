

## Infer Depth Anything on Waymo
CUDA_VISIBLE_DEVICES=0 python infer_depth.py --img-path data/waymo/training/226/images --save_npy
CUDA_VISIBLE_DEVICES=1 python infer_depth.py --img-path data/waymo/training/232/images --save_npy

CUDA_VISIBLE_DEVICES=3 python infer_depth.py --img-path data/waymo/training/237/images --save_npy
CUDA_VISIBLE_DEVICES=4 python infer_depth.py --img-path data/waymo/training/241/images --save_npy
CUDA_VISIBLE_DEVICES=5 python infer_depth.py --img-path data/waymo/training/245/images --save_npy
CUDA_VISIBLE_DEVICES=6 python infer_depth.py --img-path data/waymo/training/246/images --save_npy
CUDA_VISIBLE_DEVICES=7 python infer_depth.py --img-path data/waymo/training/271/images --save_npy