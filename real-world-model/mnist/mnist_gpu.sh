#!/bin/bash

# change conda environment
source activate tf1.15

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

platform="gpu"
name="mlp-mnist-"${platform}
datasetpath="/disk/Datasets/MNIST"
bs=64
outpath=${currentDir}/../output/mlp_${platform}_mnist
mkdir -p $outpath

echo "running model: " ${name}

time python mnist.py --platform=${platform} \
              --batch_size=${bs} \
              --output_dir=${outpath} \
              --train_dir=${datasetpath}"/train-images-idx3-ubyte" \
              --train_label=${datasetpath}"/train-labels-idx1-ubyte" \
              1>$outpath/$name.out 2>$outpath/$name.err