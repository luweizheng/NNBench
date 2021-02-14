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

data_type="mix"
platform="npu"
cnnblock='bottleneck'
outpath=${currentDir}/../output/cnn_${platform}_${data_type}
mkdir -p $outpath

for filters in 64 #32 64
do
for nblock in 2 #2 3 4 5 6 7 8
do
for input_size in 224 #200 300
do
for output_size in 1000 #1000 1500
do 
for bs in 64 128 256 # 512 1024
do

name=block_${nblock}-filtersz_${filters}-input_${input_size}-output_${output_size}-bs_${bs}
echo "running model: " $name

python3 cnn.py --platform=${platform} --data_type=${data_type} \
              --block_fn=${cnnblock} --filters=${filters} \
              --resnet_layers=${nblock},${nblock},${nblock},${nblock}\
              --input_size=${input_size} --output_size=${output_size}\
              --batch_size=${bs} --train_steps=300 \
              --output_dir=${outpath} \
              1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
done
done