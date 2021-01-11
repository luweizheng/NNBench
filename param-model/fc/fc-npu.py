'''
Fully-connected models.

@author Weizheng Lu
'''

import tensorflow as tf

from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

import time
import numpy as np
import os


tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("iterations", 100,
                        "Number of iterations per training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (accelerator chips).")
tf.flags.DEFINE_string("machine", 'npu/2x2', "machine")

tf.flags.DEFINE_integer("train_steps", 100, "Total number of steps.")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training.")
tf.flags.DEFINE_string("mode", "train", "train of infer.")
tf.flags.DEFINE_integer("warmup_steps", 300, "warmup steps")
tf.flags.DEFINE_integer("input_size", 1000, "input_size")
tf.flags.DEFINE_integer("output_size", 100, "output_size")
tf.flags.DEFINE_integer("layer", 4, "number of hidden layers")
tf.flags.DEFINE_integer("node", 1024, "number of nodes per hidden layer")
tf.flags.DEFINE_integer("mkl_threads", 1, "Just for logging purpose.")
tf.flags.DEFINE_integer("intra_threads", 0, "")
tf.flags.DEFINE_integer("inter_threads", 0, "")
tf.flags.DEFINE_string("optimizer", "rms", "Choose among rms, sgd, and momentum.")
tf.flags.DEFINE_string("data_type", "float32", "")

FLAGS = tf.flags.FLAGS

input_size = FLAGS.input_size
output_size = FLAGS.output_size
batch_size = FLAGS.batch_size
train_steps = FLAGS.train_steps
layer = FLAGS.layer
node = FLAGS.node

opt = FLAGS.optimizer
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def get_input_fn(input_size, output_size):
    '''
    Randomly genernate input dataset and labels 
    '''
    def input_fn(params):
        if FLAGS.data_type == 'float32':
            tf.logging.info("Using float32.")
            inputs = tf.random_uniform(
                [batch_size, input_size], minval=-0.5, maxval=0.5, dtype=tf.float32)
        elif FLAGS.data_type == 'float16':
            tf.logging.info("Using float16.")
            inputs = tf.random_uniform(
                [batch_size, input_size], minval=-0.5, maxval=0.5, dtype=tf.float16)

        labels = tf.random_uniform(
            [batch_size], maxval=output_size, dtype=tf.int32) 

        return inputs, labels
    return input_fn

def model_fn(features, labels, mode, params):
    net = features

    if FLAGS.data_type == 'float32':
        for i in range(layer):
            net = tf.layers.dense(
                inputs=net,
                units=node,
                name='fc_' + str(i),
                activation=tf.nn.relu)
        net = tf.layers.dense(
            inputs=net,
            units=output_size,
            name='fc_' + str(layer),
            activation=None)
  
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'logits': net,
        }
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
            )
  
    onehot_labels=tf.one_hot(labels, output_size)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=net)
  
    learning_rate = 0.1
    if opt == 'sgd':
        tf.logging.info('Using SGD optimizer')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
    elif opt == 'momentum':
        tf.logging.info('Using Momentum optimizer')
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9)
    elif opt == 'rms':
        tf.logging.info('Using RMS optimizer')
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            RMSPROP_DECAY,
            momentum=RMSPROP_MOMENTUM,
            epsilon=RMSPROP_EPSILON)
    optimizer = NPUDistributedOptimizer(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    param_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=ProfileOptionBuilder.trainable_variables_parameter())
    fl_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation())

    # eval
    if mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(labels, logits):
            predictions = tf.argmax(logits, axis=1)
            top_1_accuracy = tf.metrics.accuracy(labels, predictions)
            return {
                'Top-1 accuracy': top_1_accuracy,
            }
        eval_metrics = (metric_fn, [labels, net])

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            host_call=None,
            eval_metrics=eval_metrics)

    # train
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)

ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    print('TensorFlow version: ' + str(tf.__version__))

    # for k,v in iter(tf.app.flags.FLAGS.flag_values_dict().items()):
    #     print("***%s: %s" % (k, v))

    session_config = tf.ConfigProto()
    
    run_config = NPURunConfig(
        model_dir="./fc_model",
        session_config=session_config,
        save_checkpoints_secs=None
    )

    estimator = NPUEstimator(
        model_fn=model_fn,
        params={"output_size": output_size, "input_size": input_size, "batch_size": batch_size},
        model_dir='./fc_model',
        config=run_config
    )

    if FLAGS.mode == 'train':
        tf.logging.info('Running estimator.train()')
        estimator.train(input_fn=get_input_fn(input_size, output_size), max_steps=FLAGS.warmup_steps)
        start = time.time()
        estimator.train(input_fn=get_input_fn(input_size, output_size), max_steps=FLAGS.train_steps)
    else:
        tf.logging.info('Running estimator.predict()')
        estimator.train(input_fn=get_input_fn(input_size, output_size), max_steps=1)
        p = estimator.evaluate(input_fn=get_input_fn(input_size, output_size), steps=FLAGS.warmup_steps)
        start = time.time()
        p = estimator.evaluate(input_fn=get_input_fn(input_size, output_size), steps=FLAGS.train_steps)

    total_time = time.time() - start
    example_per_sec = batch_size * train_steps / total_time
    global_step_per_sec = train_steps / total_time
    print("Total time: " + str(total_time))
    #tf.logging.info("global_step/sec: %s" % global_step_per_sec)
    #tf.logging.info("examples/sec: %s" % example_per_sec)
    flops = batch_size * (node * node * (layer-1) + node * input_size + node * output_size)
    if FLAGS.mode == 'train':
        # Forward 2x and Backward 4x
        flops *= 6 * train_steps
    else:
        flops *= 2 * train_steps
    print('FLOPS: {}'.format(flops))
    print('TFLOPS: {}'.format(flops / total_time / 1e12))

if __name__ == '__main__':
    tf.app.run()
