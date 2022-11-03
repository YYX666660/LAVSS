#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id -2mix-LogFreq-resnet18dilated-unet7-linear-frames3stride20-maxpool-ratio-weightedLoss-channels32-epoch50-step40_60 "
OPTS+="--list_val data/test_solo.csv "
OPTS+="--ckpt ./ckpt3 "


# Models
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 0 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 1 "
# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "    # 改
OPTS+="--stride_frames 20 "     # 改
OPTS+="--frameRate 8 "

# audio-related
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

python -u train_solo_2.5Dsep_pos.py $OPTS
