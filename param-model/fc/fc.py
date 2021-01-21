'''
Fully-connected models.

@author Weizheng Lu
'''

import tensorflow as tf
import time
import numpy as np
import os

tf.flags.DEFINE_string("platform", "gpu", "which computing platform we are using, GPU/NPU")
tf.flags.DEFINE_string("output_dir", "./", "All the output data should be written into this folder.")
tf.flags.DEFINE_integer("iterations", 100,
                        "Number of iterations per training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (accelerator chips).")
tf.flags.DEFINE_string("machine", 'npu/2x2', "machine")

tf.flags.DEFINE_integer("train_steps", 100, "Total number of steps.")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training.")
tf.flags.DEFINE_string("mode", "train", "train or inference.")
tf.flags.DEFINE_integer("warmup_steps", 300, "warmup steps")
tf.flags.DEFINE_integer("input_size", 1000, "input_size")
tf.flags.DEFINE_integer("output_size", 100, "output_size")
tf.flags.DEFINE_integer("layer", 4, "number of hidden layers")
tf.flags.DEFINE_integer("nodes_per_layer", 1024, "number of nodes per hidden layer")

tf.flags.DEFINE_string("optimizer", "rms", "Choose among rms, sgd, and momentum.")
tf.flags.DEFINE_string("data_type", "float32", "float32 or mix precision")

FLAGS = tf.flags.FLAGS

input_size = FLAGS.input_size
output_size = FLAGS.output_size
batch_size = FLAGS.batch_size
train_steps = FLAGS.train_steps
layer = FLAGS.layer
nodes_per_layer = FLAGS.nodes_per_layer

if FLAGS.platform == "npu":
    from npu_bridge.estimator import npu_ops
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

opt = FLAGS.optimizer
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def get_input_fn(input_size, output_size):
    '''
    Randomly genernate input dataset and labels 
    '''
    def input_fn(params):
        tf.logging.info("Using float32.")
        inputs = tf.random_uniform(
            [batch_size, input_size], minval=-0.5, maxval=0.5, dtype=tf.float32)

        labels = tf.random_uniform(
            [batch_size], maxval=output_size, dtype=tf.int32) 

        return inputs, labels
    return input_fn

def model_fn(features, labels, mode, params):
    net = features
    for i in range(layer):
        net = tf.layers.dense(
            inputs=net,
            units=nodes_per_layer,
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
    if FLAGS.platform == "npu":
        optimizer = NPUDistributedOptimizer(optimizer)
    elif FLAGS.platform == "gpu" :
        # mixed precision
        if FLAGS.data_type == "mix":
            optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    
    tf.logging.info('enter tf.profiler function')
    
    param_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=ProfileOptionBuilder.trainable_variables_parameter())
    fl_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation())

    # eval mode
    # if mode == tf.estimator.ModeKeys.EVAL:
    #     def metric_fn(labels, logits):
    #         predictions = tf.argmax(logits, axis=1)
    #         top_1_accuracy = tf.metrics.accuracy(labels, predictions)
    #         return {
    #             'Top-1 accuracy': top_1_accuracy,
    #         }
    #     eval_metrics = (metric_fn, [labels, net])

    #     return tf.estimator.EstimatorSpec(
    #         mode=mode,
    #         loss=loss,
    #         train_op=train_op,
    #         host_call=None,
    #         eval_metrics=eval_metrics)

    # print loss every n iterations
    train_hook = tf.estimator.LoggingTensorHook(
            tensors={"loss" : loss},
            every_n_iter=10
        )
    # train
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[train_hook])

ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
def main(unused_argv):
    del unused_argv
    tf.logging.set_verbosity(tf.logging.INFO)
    print('\nTensorFlow version: ' + str(tf.__version__))

    # for k,v in iter(tf.app.flags.FLAGS.flag_values_dict().items()):
    #     print("***%s: %s" % (k, v))
    if FLAGS.platform == "npu":
        session_config = tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=False)
    elif FLAGS.platform == "gpu":
        # with `allow_soft_placement=True` TensorFlow will automatically help us choose a device in case the specific one does not exist
        # with `log_divice_placement=True` we can see all the operations and tensors are mapped to which device
        session_config = tf.ConfigProto(allow_soft_placement=True, 
            log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True))
    model_dir = os.path.join(FLAGS.output_dir, "model_dir", "fc",
                '-'.join(
                    ["layer_" + str(FLAGS.layer), "nodes_" + str(FLAGS.nodes_per_layer), 
                    "input_" + str(FLAGS.input_size), "output_" + str(FLAGS.output_size),
                    "bs_" + str(FLAGS.batch_size)]))

    if FLAGS.platform == "npu":
        run_config = NPURunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=None,
            precision_mode="allow_mix_precision"
        )
    elif FLAGS.platform == "gpu":
        # mixed precision
        if FLAGS.data_type == "mix":
            session_config.graph_options.rewrite_options.auto_mixed_precision=1
        run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=None
        )

    if FLAGS.platform == "npu":
        estimator = NPUEstimator(
            model_fn=model_fn,
            params={"output_size": output_size, "input_size": input_size, "batch_size": batch_size},
            model_dir=model_dir,
            config=run_config
        )
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params={"output_size": output_size, "input_size": input_size, "batch_size": batch_size},
            model_dir=model_dir,
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
    flops = batch_size * (nodes_per_layer * nodes_per_layer * (layer-1) + 
            nodes_per_layer * input_size + nodes_per_layer * output_size)
    if FLAGS.mode == 'train':
        # Forward 2x and Backward 4x
        flops *= 6 * train_steps
    else:
        flops *= 2 * train_steps
    print('FLOPS: {}'.format(flops))
    print('TFLOPS: {}'.format(flops / total_time / 1e12))

if __name__ == '__main__':
    tf.app.run()
