#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#export VERBOSE=1
export DMLC_INTERFACE=ib0
export MXNET_CPU_WORKER_NTHREADS=16
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cudnn/lib64
LAUNCH_BIN="python /data/home/bixi/source/mxnet_dmlc/tools/launch.py -n 2 -s 6 --launcher ssh -H ${DIR}/hosts"
SYNC_DST_DIR="/tmp/mxnet"
NETWORK=xception_aligned_x38_v4
GPUS=0,1,2,3,4,5,6,7
KV_STORE=dist_sync
NUM_EPOCHS=240
LR=0.9
LR_FACTOR=0.94
GOOGLE_LR_SCHEDULER=1
GOOGLE_LR_DECAY_STEPS=2
BATCH_SIZE=256
DISP_BATCHES=200
MODEL_PREFIX=$DIR/model
TOP_K=5
LOG_DIR=$DIR/log
IMAGE_SHAPE=3,224,224
WD=0.00004
MOM=0.9
BN_MOM=0.99

#RANDOM_CROP=1
#RANDOM_MIRROR=1
#RANDOM_RESIZE=1
#BRIGHTNESS=0.4
#CONTRAST=0.4
#SATURATION=0.4
#PCA_NOISE=0.1
#INTER_METHOD=3

# Use torch aspect method
ASPECT_RATIO_METHOD=1
MIN_RANDOM_SCALE=0.224
MAX_RANDOM_ASPECT_RATIO=0.25
MAX_RANDOM_H=36
MAX_RANDOM_S=50
MAX_RANDOM_L=50


#DATA_TRAIN_DIR=/data/home/bixi/data/ILSVRC2012_CLS_MXNET/train/raw_q95_chunks_40/
#
#DATA_TRAIN=${DATA_TRAIN_DIR}train_0.rec
#for ((i=1; i<40; i++)); do
#	DATA_TRAIN=${DATA_TRAIN}";"${DATA_TRAIN_DIR}train_${i}.rec
#done

DATA_TRAIN=/data/home/bixi/data/ILSVRC2012_CLS_MXNET/train/raw_q95/train.rec
DATA_TRAIN_IDX=/data/home/bixi/data/ILSVRC2012_CLS_MXNET/train/raw_q95/train.idx
DATA_VAL=/data/home/bixi/data/ILSVRC2012_CLS_MXNET/val/256_q95/val.rec

mkdir -p $DIR/log
cd /data/home/bixi/source/mx_rfcn_leoxiaobin/image-classification
${LAUNCH_BIN} /data/home/bixi/anaconda2/bin/python train_imagenet.py --network ${NETWORK} \
    --gpus ${GPUS} \
    --kv-store ${KV_STORE} \
    --num-epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --lr-factor ${LR_FACTOR} \
    --google-lr-scheduler ${GOOGLE_LR_SCHEDULER} \
    --google-lr-decay-steps ${GOOGLE_LR_DECAY_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --disp-batches ${DISP_BATCHES} \
    --model-prefix ${MODEL_PREFIX} \
    --top-k ${TOP_K} \
    --log-dir ${LOG_DIR} \
    --image-shape ${IMAGE_SHAPE} \
    --wd ${WD} \
    --mom ${MOM} \
    --data-train ${DATA_TRAIN} \
    --data-val ${DATA_VAL} \
    --bn-mom ${BN_MOM} \
    --optimizer nag \
    --data-train ${DATA_TRAIN} \
    --data-val ${DATA_VAL} \
    --save-state 0 \
    --sync-dst-dir ${SYNC_DST_DIR} \
    --min-random-scale ${MIN_RANDOM_SCALE} \
    --max-random-h ${MAX_RANDOM_H} \
    --max-random-s ${MAX_RANDOM_S} \
    --max-random-l ${MAX_RANDOM_L} \
    --aspect-ratio-method ${ASPECT_RATIO_METHOD} \
    --max-random-aspect-ratio ${MAX_RANDOM_ASPECT_RATIO} \
    #--data-train-idx ${DATA_TRAIN_IDX} \
    #--random-crop ${RANDOM_CROP} \
    #--random-mirror ${RANDOM_MIRROR} \
    #--random-resize ${RANDOM_RESIZE} \
    #--brightness ${BRIGHTNESS} \
    #--contrast ${CONTRAST} \
    #--saturation ${SATURATION} \
    #--pca-noise ${PCA_NOISE} \
    #--inter-method ${INTER_METHOD} \
