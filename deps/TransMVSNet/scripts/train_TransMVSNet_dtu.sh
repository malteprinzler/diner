#!/bin/bash

MVS_TRAINING="data/DTU"                    # path to dataset
LOG_DIR="outputs/dtu/TransMVSNet_training" # path to log dir
NGPUS=8
BATCH_SIZE=1

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node=$NGPUS deps/TransMVSNet/train.py \
  --logdir=$LOG_DIR \
  --dataset=dtu_yao \
  --batch_size=$BATCH_SIZE \
  --epochs=20 \
  --trainpath=$MVS_TRAINING \
  --trainlist=deps/TransMVSNet/lists/dtu/train.txt \
  --testlist=deps/TransMVSNet/lists/dtu/val.txt \
  --numdepth=192 \
  --ndepths="48,32,8" \
  --nviews=4 \
  --wd=0.0001 \
  --depth_inter_r="4.0,1.0,0.5" \
  --lrepochs="7,10,15:2" \
  --dlossw="1.0,1.0,1.0"
