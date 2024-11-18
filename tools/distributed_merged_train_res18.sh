#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/1114_one_stage_res18
CONFIG=projects/configs/merged_subset_A_R18.py

GPUS=$1
PORT=${PORT:-28560}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/merged_train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log