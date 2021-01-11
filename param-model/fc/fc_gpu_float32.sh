#!/bin/bash

# change conda environment
source activate tf1.15

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

data_type="float32"
platform="gpu"
outpath=${currentDir}/../output/fc_${platform}_${data_type}
mkdir -p $outpath

for layer in 16 32 64 128 256
do
for nodes_per_layer in 128 256 512 1024 2048 4096 8192
do
for input in 256 512 1024
do
for output in 256 512 1024
do
for batch_size in 64 128 256 512 1024 2048 4096 8192
do

name=layer_${layer}-nodes_${nodes_per_layer}-input_${input}-output_${output}-bs_${batch_size}
echo $name

python fc.py --platform=${platform} --data_type=${data_type} --layer=${layer} \
            --nodes_per_layer=${nodes_per_layer} --input_size=${input} --output_size=${output} \
            --batch_size=${batch_size} --train_steps=100 \
            --output_dir=${outpath} \
            1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
done
done