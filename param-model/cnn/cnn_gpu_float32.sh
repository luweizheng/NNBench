#!/bin/bash

# change conda environment
source activate tf1.15

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

data_type="float32"
platform="gpu"
cnnblock='bottleneck'
outpath=${currentDir}/../output/${cnnblock}_${platform}_${data_type}
mkdir -p $outpath

input_size=224
output_size=1000

for filters in 16 #32 64
do
for nblock in 1 #2 3 4 5 6 7 8
do
for input_size in 224 #200 300
do
for output_size in 1000 #1000 1500
do 
for bs in 64 # 128 256 512 1024
do

name=block_${nblock}-filtersz_${filters}-input_${input_size}-output_${output_size}-bs_${bs}
echo "running model: " $name

# skip the experiment if its performance report exists
grep "examples/sec" $outpath/$name.err > tmp
filesize=$(stat -c%s tmp)
if [ "${filesize}" -gt 0 ];
then
  echo "skipping "$name
  continue
fi

python cnn.py --platform=${platform} --data_type=${data_type} \
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