CUDA_VISIBLE_DEVICES=0 python evaluate_zero_shot_cls.py --model_name our_medclip --prompt_style gloria \
    --dataset_list mimic_5x200 chexpert_5x200 chexpert nih padchest --batch_size 128 \
    --ckpt_path /home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_30_12_11_00/epoch=8-step=3780.ckpt