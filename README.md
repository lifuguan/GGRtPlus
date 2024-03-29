

## Infer Depth Anything on Waymo
CUDA_VISIBLE_DEVICES=0 python infer_depth.py --img-path data/waymo/training/297/images --save_npy
CUDA_VISIBLE_DEVICES=1 python infer_depth.py --img-path data/waymo/training/302/images --save_npy
CUDA_VISIBLE_DEVICES=3 python infer_depth.py --img-path data/waymo/training/312/images --save_npy
CUDA_VISIBLE_DEVICES=4 python infer_depth.py --img-path data/waymo/training/314/images --save_npy
CUDA_VISIBLE_DEVICES=5 python infer_depth.py --img-path data/waymo/training/362/images --save_npy
CUDA_VISIBLE_DEVICES=6 python infer_depth.py --img-path data/waymo/training/482/images --save_npy
CUDA_VISIBLE_DEVICES=7 python infer_depth.py --img-path data/waymo/training/495/images --save_npy

CUDA_VISIBLE_DEVICES=0 python infer_depth.py --img-path data/waymo/training/524/images --save_npy
CUDA_VISIBLE_DEVICES=1 python infer_depth.py --img-path data/waymo/training/527/images --save_npy
CUDA_VISIBLE_DEVICES=2 python infer_depth.py --img-path data/waymo/training/753/images --save_npy
CUDA_VISIBLE_DEVICES=3 python infer_depth.py --img-path data/waymo/training/780/images --save_npy