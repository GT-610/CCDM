@echo off

@REM set CUDA_VISIBLE_DEVICES=0

set DATA_NAME="UTKFace"

set ROOT_PATH=<YOUR_PATH>/CCDM/CCDM_unified
set DATA_PATH=<YOUR_PATH>/datasets/%DATA_NAME%

set SETTING="Setup_CCDM"
set SIGMA=-1.0
set KAPPA=-1.0
set TYPE="hard"

python main.py ^
    --setting_name %SETTING% ^
    --root_path %ROOT_PATH% --data_name %DATA_NAME% --data_path %DATA_PATH% ^
    --image_size 64 --train_amp ^
    --min_label 1 --max_label 60 --num_img_per_label_after_replica 200 ^
    --pred_objective pred_x0 ^
    --model_channels 72 --cond_drop_prob 0.1 --channel_mult 1_2_4_4_8 ^
    --y2h_embed_type "resnet" --y2cov_embed_type "resnet" --use_Hy ^
    --niters 100000 --resume_niter 0 --train_lr 1e-4 --train_timesteps 1000 ^
    --train_batch_size 128 --gradient_accumulate_every 1 ^
    --kernel_sigma %SIGMA% --threshold_type %TYPE% --kappa %KAPPA% ^
    --sample_every 10000 --save_every 10000 ^
    --sample_timesteps 250 --sample_cond_scale 1.5 ^
    --sampler ddim --samp_batch_size 200 --nfake_per_label 1000 ^
    --dump_fake_data ^ %*