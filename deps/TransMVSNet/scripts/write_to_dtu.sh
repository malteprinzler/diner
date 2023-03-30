#!/bin/bash
export PYTHONPATH="deps/TransMVSNet:$PYTHONPATH"

DATA_ROOT="data/DTU/" # path to processed facescape dataset
OUTDEPTHNAME="TransMVSNet"  # prefix of the output depth files
LOG_DIR="outputs/dtu/TransMVSNet_writing"
CKPT="assets/ckpts/dtu/TransMVSNet.ckpt"  # path to pretrained checkpoint
NGPUS=1
BATCH_SIZE=1

if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi
python deps/TransMVSNet/train.py \
  --mode="write_prediction" \
  --loadckpt=$CKPT \
	--logdir=$LOG_DIR \
	--dataset=dtu_yao \
	--batch_size=$BATCH_SIZE \
	--trainpath=$DATA_ROOT \
	--trainlist=deps/TransMVSNet/lists/dtu/train.txt \
	--testlist=deps/TransMVSNet/lists/dtu/val.txt \
	--numdepth=192 \
	--ndepths="48,32,8" \
	--nviews=4 \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="7,10,15:2" | tee -a $LOG_DIR/log.txt