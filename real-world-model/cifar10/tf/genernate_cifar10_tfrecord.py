"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.
Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

import tensorflow as tf
import argparse
import pickle
import os
 
def unpickle(file):
    with open(file,'rb') as f:
      dict = pickle.load(f,encoding='bytes')
    return dict
 
def _int64_feature(value):
	  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
	  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
 
def restore(inputfilename, outputfilename):
    dict = unpickle(inputfilename)
    # dict: include [b'batch_label', b'labels', b'data', b'filenames']
    # we just choose labels and data. And we choose restore it by int64
    # labels:[10000,1]
    labels= dict[b'labels']
    #images:[10000,3072]
    images = dict[b'data']
    writer = tf.python_io.TFRecordWriter(outputfilename)
    for i in range(10000):
        image_raw = images[i].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
          'raw_image':_bytes_feature(image_raw),
          'label':_int64_feature(labels[i])
        }))
        writer.write(example.SerializeToString())


def main(data_dir):
    train_filenames = [os.path.join(data_dir,'data_batch_%d' %i) for i in range(1,6)]
    for filename in train_filenames:
        restore(filename,filename+'.tfrecord')
    eval_filename = os.path.join(data_dir,'test_batch')
    restore(eval_filename, os.path.join(data_dir, "eval.tfrecord"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default='',
        help='Directory to download and extract CIFAR-10 to.')

    args = parser.parse_args()
    main(args.data_dir)
