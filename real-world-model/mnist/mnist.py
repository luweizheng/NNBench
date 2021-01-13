'''
MLP model based on MNIST dataset.

@author Weizheng Lu
'''

import tensorflow as tf
import time
import numpy as np
import os

LEARNING_RATE = 1e-4

_TENSORS_TO_LOG = dict((x, x) for x in [
    'learning_rate',
    'cross_entropy',
    'train_accuracy'
])

# whether use NPU or not
tf.flags.DEFINE_string("platform", "gpu", "which computing platform we are using, GPU/NPU")
tf.flags.DEFINE_string("output_dir", "./", "All the output data should be written into this folder.")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training.")
tf.flags.DEFINE_string("train_dir", None, "training data path.")
tf.flags.DEFINE_string("train_label", None, "training label path.")
tf.flags.DEFINE_string("mode", "train", "train or inference.")

FLAGS = tf.flags.FLAGS

if FLAGS.train_dir is None or FLAGS.train_label is None:
    tf.logging.info('train_dir or train_label is None, cannot train model, exit.')
    exit()

if FLAGS.platform == "npu":
    from npu_bridge.estimator import npu_ops
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

def create_model():
    input_shape = [28, 28, 1]
    l = tf.keras.layers
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(28 * 28,)
            ),
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dense(10)
        ]
    )

def model_fn(features, labels, mode, params):
    model = create_model()

    image = features
    if isinstance(image, dict):
        image = features['image']
    
    param_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=ProfileOptionBuilder.trainable_variables_parameter())
    fl_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation())

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        if FLAGS.platform == "npu":
            optimizer = NPUDistributedOptimizer(optimizer)

        logits = model(image, training=True)
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels,
                                                                logits=logits)
        
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1)
        )

        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.identity(accuracy[1], name='train_accuracy')

        tf.summary.scalar('train_accuracy', accuracy[1])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(
                loss,
                tf.compat.v1.train.get_or_create_global_step()
            )
        )

def dataset():
    def decode_image(image):
        image = tf.io.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0
    
    def decode_label(label):
        label = tf.io.decode_raw(label, tf.uint8)
        label = tf.reshape(label, [])
        return tf.cast(label, tf.int32)

    # load training data and labels
    images = tf.data.FixedLengthRecordDataset(
            FLAGS.train_dir, 28 * 28, header_bytes=16
        ).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
            FLAGS.train_label, 1, header_bytes=8
        ).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def train_input_fn():
    with tf.name_scope('data_input'):
        ds = dataset()
        ds = ds.shuffle(buffer_size=50000).batch(FLAGS.batch_size, drop_remainder=True)
        ds = ds.repeat(1)
        return ds

train_hook = tf.estimator.LoggingTensorHook(
    tensors={'cross_entropy': 'cross_entropy', 'accuracy': 'train_accuracy'},
    every_n_iter=100
)

ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
def main(unused_argv):
    del unused_argv

    print('\nTensorFlow version: ' + str(tf.__version__))

    if FLAGS.platform == "npu":
        session_config = tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=False)
    elif FLAGS.platform == "gpu":
        # with `allow_soft_placement=True` TensorFlow will automatically help us choose a device in case the specific one does not exist
        # with `log_divice_placement=True` we can see all the operations and tensors are mapped to which device
        session_config = tf.ConfigProto(allow_soft_placement=True, 
            log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True))

    model_dir = os.path.join(FLAGS.output_dir, "model_dir",
                    '-'.join(["mlp", "mnist", FLAGS.platform]))

    if FLAGS.platform == "npu":
        run_config = NPURunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=None,
            #precision_mode="allow_mix_precision"
        )
    else:
        run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=None,
            #precision_mode="allow_mix_precision"
        )

    if FLAGS.platform == "npu":
        estimator = NPUEstimator(
            model_fn=model_fn,
            # params={"output_size": output_size, "input_size": input_size, "batch_size": batch_size},
            model_dir=model_dir,
            config=run_config
        )
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            # params={"output_size": output_size, "input_size": input_size, "batch_size": batch_size},
            model_dir=model_dir,
            config=run_config
        )

    tf.logging.info('Running estimator.train()')
    with tf.name_scope('train'):
        estimator.train(input_fn=train_input_fn, hooks=[train_hook])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()