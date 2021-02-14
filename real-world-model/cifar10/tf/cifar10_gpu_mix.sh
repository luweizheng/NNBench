#!/bin/bash

# change conda environment
source activate tf1.15

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

platform="gpu"
data_type="mix"
name="cifar10-"${platform}-${data_type}
datasetpath="/disk/Datasets/CIFAR10/TFRecord"
bs=128
outpath=${currentDir}/../output/${name}/${currtime}
mkdir -p $outpath

echo "running model: " ${name}

start=`date +%s`
python cifar10-cnn.py --platform=${platform} \
              --train_batch_size=${bs} \
              --output_dir=${outpath} \
              --train_dir=${datasetpath} \
              --eval_dir=${datasetpath} \
              1>$outpath/$name.out 2>$outpath/$name.err

end=`date +%s`
runtime=$((end-start))
echo "total run time: "${runtime}