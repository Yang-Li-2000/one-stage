#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/1120_one_stage_smerf_swins
CONFIG=projects/configs/one_stage_smerf_swins_subset_A.py

GPUS=$1
PORT=${PORT:-28520}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train_one_stage_smerf_swins.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log