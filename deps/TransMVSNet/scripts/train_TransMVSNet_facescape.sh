#!/bin/bash

MVS_TRAINING="data/FACESCAPE_PROCESSED"          # path to dataset mvs_training
LOG_DIR="outputs/facescape/TransMVSNet_training" # path to checkpoints
NGPUS=8
BATCH_SIZE=1

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node=$NGPUS deps/TransMVSNet/train.py \
  --logdir=$LOG_DIR \
  --dataset=facescape \
  --batch_size=$BATCH_SIZE \
  --epochs=20 \
  --trainpath=$MVS_TRAINING \
  --numdepth=384 \
  --ndepths="96,64,16" \
  --nviews=4 \
  --wd=0.0001 \
  --depth_inter_r="4.0,1.0,0.5" \
  --lrepochs="1,2,3:2" \
  --dlossw="1.0,1.0,1.0"
