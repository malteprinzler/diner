#!/bin/bash
source venv/bin/activate
export PYTHONPATH="deps/TransMVSNet:$PYTHONPATH"

DATA_ROOT="data/FACESCAPE_PROCESSED/" # path to processed facescape dataset
OUTDEPTHNAME="TransMVSNet"  # prefix of the output depth files
LOG_DIR="outputs/facescape/TransMVSNet_writing"
CKPT="assets/ckpts/facescape/TransMVSNet.ckpt"  # path to pretrained checkpoint
NGPUS=1
BATCH_SIZE=1

if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi
python -m torch.distributed.launch --nproc_per_node=$NGPUS deps/TransMVSNet/train.py \
  --mode="write_prediction" \
  --outdepthname=$OUTDEPTHNAME \
  --maskoutput \
  --loadckpt=$CKPT \
	--logdir=$LOG_DIR \
	--dataset=facescape \
	--batch_size=$BATCH_SIZE \
	--trainpath=$DATA_ROOT \
	--numdepth=384 \
	--ndepths="96,64,16" \
	--nviews=4 \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="7,10,15:2" | tee -a $LOG_DIR/log.txt
