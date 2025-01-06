#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/0106_one_stage_smerf_distill
CONFIG=projects/configs/distillation.py

GPUS=$1
PORT=${PORT:-28520}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train_one_stage_smerf.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log