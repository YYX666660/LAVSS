#!/bin/bash
OPTS=""
OPTS+="--id LAVSS "
OPTS+="--list_train ./data/train.csv "
OPTS+="--list_val ./data/val.csv "
OPTS+="--ckpt ./ckpt "     # added
# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
# Load model path
OPTS+="--weights_visual ckpt_2.5Dsep_musicpre/-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride20-maxpool-ratio-weightedLoss-channels32-epoch60-step40_50/frame_best.pth "
OPTS+="--weights_unet ckpt_2.5Dsep_musicpre/-2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride20-maxpool-ratio-weightedLoss-channels32-epoch60-step40_50/sound_best.pth "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 1 "
OPTS+="--loss bce "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "
# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 24 "
OPTS+="--frameRate 8 "
# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

# learning params
# OPTS+="--num_gpus 4 "
OPTS+="--workers 12 "       
OPTS+="--batch_size_per_gpu 32 "
OPTS+="--lr_frame 1e-4 "
OPTS+="--lr_sound 1e-3 "
OPTS+="--lr_synthesizer 1e-3 "
OPTS+="--num_epoch 50 "
OPTS+="--lr_steps 40 80 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

python -u train_solo_2.5Dsep_pos.py $OPTS