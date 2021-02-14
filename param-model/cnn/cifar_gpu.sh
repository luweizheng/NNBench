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

for filters in 16
do
for nblock in 2 # 3 4 5
do
for input_size in 32 #200 300
do
for output_size in 10 #1000 1500
do 
for bs in 128 # 256 512 1024
do

name=block_${nblock}-filtersz_${filters}-input_${input_size}-output_${output_size}-bs_${bs}
echo "running model: " $name

python3 cifar.py --platform=${platform} --data_type=${data_type} \
              --block_fn=${cnnblock} --filters=${filters} \
              --resnet_layers=${nblock},${nblock},${nblock},${nblock} \
              --input_size=${input_size} --output_size=${output_size}\
              --batch_size=${bs} --train_steps=300 \
              --output_dir=${outpath} \
              1>$outpath/$name.out 2>$outpath/$name.err

done
done
done
done
done