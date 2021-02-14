#!/bin/bash

# change conda environment
source activate tf1.15

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

data_type="float32"
platform="gpu"
cnnblock='residual'
outpath=${currentDir}/../output/${cnnblock}_${platform}_${data_type}
mkdir -p $outpath

name=block_${nblock}-filtersz_${filters}-input_${input_size}-output_${output_size}-bs_${bs}
echo "running model: " $name

python3 cifar.py --platform=${platform} --data_type=${data_type} \
              --block_fn=${cnnblock} --filters=16 \
              --resnet_layers=2,2,2 \
              --train_dir="/disk/Datasets/CIFAR10/TFRecord" \
              --batch_size=128 --train_steps=300 \
              --output_dir=${outpath} \
              1>$outpath/$name.out 2>$outpath/$name.err