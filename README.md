# DCXFM
Demo code of "Scaling Chest X-ray Foundation Models from Mixed Supervisions for Dense Prediction".

### Installation
```
pip install -r requirements.txt
# For cuda 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# For cuda 12.1
pip3 install torch torchvision torchaudio
pip instal -e .
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Dataset preprocessing

All preprocessing code is available in `dcxfm/preprocess`.

### Pretraining

Run SDMP:
```
cd scrpits/
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_sdmp.py --num_devices 4 --use_i2t_loss \
    --loss_type soft_cont --use_local_loss --use_self_distil_loss \
    --train_data_pct 0.2 --lambda3 2
```

Learning from Dense Annotations:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 4 --config-file ../configs/vitb_r101_384.yaml SEED 5
```

### Evaluation

Please check `scripts/run_phrase_grounding.sh`, `scripts/run_seg.sh`, `scripts/run_cls.sh`.

TODO:
- [ ] update benchmark description