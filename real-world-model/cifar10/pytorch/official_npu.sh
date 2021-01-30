#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/:/usr/local/python3.7.5/lib/:/usr/local/openblas/lib:/usr/local/lib/:/usr/lib64/:/usr/lib/:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/add-ons/:/usr/lib/aarch64_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/Ascend/nnae/latest/fwkacllib/ccec_compiler/bin/:/usr/local/Ascend/nnae/latest/toolkit/tools/ide_daemon/bin/
export ASCEND_OPP_PATH=/usr/local/Ascend/nnae/latest/opp/
export OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libaicpu_engine.so:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so
export PYTHONPATH=/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/:/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/schedule_search.egg:$PYTHONPATH

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

arch="resnet32"
platform="npu"
save_dir=${currentDir}"/"${arch}_${platform}"/"${currtime}
name="cifar10-"${arch}

echo "save dir:"${save_dir}
echo "running "${name}
device_id=0

python3.7 ${currentDir}/official_train.py \
        --data "~/Datasets/CIFAR10/" \
        --npu ${device_id} \
        --platform "npu" \
        --arch resnet32 \
        -j64 \
        -b128 \
        --lr 0.2 \
        --warmup 5 \
        --label-smoothing=0.0 \
        --epochs 10 \
        --optimizer-batch-size 512