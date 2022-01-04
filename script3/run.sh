
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVACT --l1_weight_grd 0 --perceptual_weight_grd 1 --skip 0 --heightPlaneNum 64 --mode test --checkpoint tt
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVUSA --l1_weight_grd 0 --perceptual_weight_grd 1 --skip 0 --heightPlaneNum 1 #--mode test --checkpoint tt

#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVUSA --l1_weight_grd 100 --perceptual_weight_grd 0 --skip 0 --heightPlaneNum 32 --mode test --checkpoint tt
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVUSA --l1_weight_grd 100 --perceptual_weight_grd 0 --skip 0 --heightPlaneNum 64 --mode test --checkpoint tt
#
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVUSA --l1_weight_grd 0 --perceptual_weight_grd 1 --skip 1 --heightPlaneNum 32 --mode test --checkpoint tt
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVUSA --l1_weight_grd 0 --perceptual_weight_grd 1 --skip 1 --heightPlaneNum 64 --mode test --checkpoint tt
#
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVUSA --l1_weight_grd 100 --perceptual_weight_grd 0 --skip 1 --heightPlaneNum 32 --mode test --checkpoint tt
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVUSA --l1_weight_grd 100 --perceptual_weight_grd 0 --skip 1 --heightPlaneNum 64 --mode test --checkpoint tt


#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVACTunaligned --l1_weight_grd 0 --perceptual_weight_grd 1 --skip 0 --heightPlaneNum 1 # --mode test --checkpoint tt
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVACTunaligned --l1_weight_grd 0 --perceptual_weight_grd 1 --skip 0 --heightPlaneNum 64 #--mode test --checkpoint tt
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVACTunaligned --l1_weight_grd 100 --perceptual_weight_grd 0 --skip 0 --heightPlaneNum 1 #--mode test --checkpoint tt
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset CVACTunaligned --l1_weight_grd 100 --perceptual_weight_grd 0 --skip 0 --heightPlaneNum 64 #--mode test --checkpoint tt



