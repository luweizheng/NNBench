#!/bin/bash

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

# set Huawei Ascend environments
export install_path=/usr/local/Ascend/
export ascend_toolkit_path=${install_path}/ascend-toolkit/latest

export LD_LIBRARY_PATH=/usr/local:/usr/local/lib/:usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${install_path}/add-ons:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${install_path}/driver/lib64/driver:${install_path}/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ascend_toolkit_path}/fwkacllib/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=${ascend_toolkit_path}/fwkacllib/python/site-packages:${ascend_toolkit_path}/fwkacllib/python/site-packages/auto_tune.egg/auto_tune:${ascend_toolkit_path}/fwkacllib/python/site-packages/schedule_search.egg:${ascend_toolkit_path}/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
export PYTHONPATH=/usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages:$PYTHONPATH

export PATH=${ascend_toolkit_path}/fwkacllib/ccec_compiler/bin:${ascend_toolkit_path}/fwkacllib/bin:$PATH

export ASCEND_OPP_PATH=${ascend_toolkit_path}/opp


export HCCL_CONNECT_TIMEOUT=600

# user env
export JOB_ID=${currtime}

device_id="6"
export DEVICE_ID=${device_id}

device_count="1"
export RANK_SIZE=1
export RANK_INDEX=0

export RANK_ID="localhost-"currtime

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8))
export DEVICE_INDEX=${DEVICE_INDEX}

platform="npu"
name="mlp-mnist-"${platform}
datasetpath="/home/luweizheng/ml/mnist_910"
bs=64
outpath=${currentDir}/../output/mlp_${platform}_mnist
mkdir -p $outpath

echo "running model: " ${name}

start=`date +%s`
python mnist.py --platform=${platform} \
              --batch_size=${bs} \
              --output_dir=${outpath} \
              --train_dir=${datasetpath}"/train-images-idx3-ubyte" \
              --train_label=${datasetpath}"/train-labels-idx1-ubyte" \
              1>$outpath/$name.out 2>$outpath/$name.err
wait
end=`date +%s`
runtime=$((end-start))
echo "total run time: "${runtime}
