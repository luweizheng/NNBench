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
export JOB_ID=123456789

device_id="0"
export DEVICE_ID=0

device_count="1"
export RANK_SIZE=1
export RANK_INDEX=0

pod_name=$3
export RANK_ID="localhost-"+currtime

DEVICE_INDEX=$(( DEVICE_ID + RANK_INDEX * 8))
export DEVICE_INDEX=${DEVICE_INDEX}

data_type="float32"
platform="npu"
outpath=${currentDir}/../output/fc_${platform}_${data_type}
mkdir -p $outpath

for layer in 16
do
for nodes_per_layer in 128 256 #512 1024 2048 4096 8192
do
for input in 512
do
for input in 512
do
for batch_size in 64 128 #256 512 1024 2048 4096 8192 16384
do

name=layer_${layer}-nodes_${nodes_per_layer}-input_${input}-output_${output}-bs_${batch_size}
echo $name

python fc.py --platform=${platform} --data_type=${data_type} --layer=${layer} \
            --nodes_per_layer=${nodes_per_layer} --input_size=${input} --output_size=${input} \
            --batch_size=${batch_size} --train_steps=100 \
            --output_dir=${outpath} \
            1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
done
done