#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/merged_toponet
CONFIG=projects/configs/merged_subset_A.py

CHECKPOINT=${WORK_DIR}/epoch_24.pth

GPUS=$1
PORT=${PORT:-28550}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/merged_test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:2} \
    2>&1 | tee ${WORK_DIR}/test.${timestamp}.log
