CUDA_VISIBLE_DEVICES=1 python evaluate_zero_shot_seg.py --model_name our_medclip \
    --ckpt_path /home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_05_05_00_03_32/epoch=8-step=3780.ckpt
    
# 20% MIMIC-CXR
CUDA_VISIBLE_DEVICES=0 python evaluate_zero_shot_seg.py --model_name our_medclip \
    --dataset_dir /disk1/fywang/CXR_dataset \
    --ckpt_path /home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_21_14_48_11/epoch=11-step=5040.ckpt

# 100% MIMIC-CXR
CUDA_VISIBLE_DEVICES=5 python evaluate_zero_shot_seg.py --model_name our_medclip \
    --dataset_dir /disk1/fywang/CXR_dataset \
    --ckpt_path /home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_18_01_22_16/epoch=8-step=18900.ckpt
    
# stage 2 model
CUDA_VISIBLE_DEVICES=3 python evaluate_zero_shot_seg.py --model_name our_medclip_s2 \
    --dataset_dir /disk1/fywang/CXR_dataset \
    --ckpt_path /home/fywang/Documents/CXRSeg/scripts/output/2024-05-04_16-38-32/model_0000199.pth