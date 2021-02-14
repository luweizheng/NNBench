'''
Fully-connected models.

@author Weizheng Lu
'''

import tensorflow as tf
import time
import numpy as np
import os
import resnet_model

# whether use NPU or not
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
tf.flags.DEFINE_integer("input_size", 224, "input_size")
tf.flags.DEFINE_integer("output_size", 1000, "output_size")
tf.flags.DEFINE_integer("filters", 4, "number of filters or channels of the image")
tf.flags.DEFINE_string("resnet_layers", "1,1,1,1", "residual blocks in each group")
tf.flags.DEFINE_string("block_fn", "residual", "choose from residual and bottleneck")

tf.flags.DEFINE_string("optimizer", "rms", "Choose among rms, sgd, and momentum.")
tf.flags.DEFINE_string("data_type", "float32", "")

FLAGS = tf.flags.FLAGS

input_size = [FLAGS.input_size, FLAGS.input_size, 3]
output_size = FLAGS.output_size
batch_size = FLAGS.batch_size
train_steps = FLAGS.train_steps
resnet_layers = [int(i) for i in FLAGS.resnet_layers.split(',')]
block_fn = FLAGS.block_fn
filters = FLAGS.filters

if FLAGS.platform == "npu":
    from npu_bridge.estimator import npu_ops
    from npu_bridge.estimator.npu.npu_config import NPURunConfig
    from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    from npu_bridge.estimator.npu.npu_config import ProfilingConfig

opt = FLAGS.optimizer
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

def get_input_fn(input_size, output_size):
    '''
    Randomly genernate input dataset and labels 
    '''
    def input_fn(params):
        batch_size = params['batch_size']
        if FLAGS.data_type == 'float32':
            dataset = tf.data.Dataset.range(1).repeat().map(
                lambda x: (tf.cast(tf.constant(np.random.random_sample(input_size).astype(np.float32), tf.float32), tf.float32),
                        tf.constant(np.random.randint(output_size, size=(1,))[0], tf.int32)))
        elif FLAGS.data_type == 'float16':
            dataset = tf.data.Dataset.range(1).repeat().map(
                lambda x: (tf.cast(tf.constant(np.random.random_sample(input_size).astype(np.float32), tf.float16), tf.float16),
                        tf.constant(np.random.randint(output_size, size=(1,))[0], tf.int32)))
        
        dataset = dataset.prefetch(batch_size)

        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))

        dataset = dataset.prefetch(4)     # Prefetch overlaps in-feed with training
        images, labels = dataset.make_one_shot_iterator().get_next()
        return images, labels
    return input_fn

def model_fn(features, labels, mode, params):
    output_size = params['output_size']
    net = features

    if FLAGS.data_type == 'float32':
        network = resnet_model.resnet_v1(
            resnet_layers,
            block_fn,
            num_classes=output_size,
            data_format='channels_last',
            filters=filters)

        net = network(inputs=features, is_training=True)
        
    onehot_labels=tf.one_hot(labels, output_size)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=net)

    learning_rate = tf.train.exponential_decay(
        0.1, tf.train.get_global_step(), 25000, 0.97)
    
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

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    param_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=ProfileOptionBuilder.trainable_variables_parameter())
    fl_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation())

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
        session_config = tf.ConfigProto()
    elif FLAGS.platform == "gpu":
        # with `allow_soft_placement=True` TensorFlow will automatically help us choose a device in case the specific one does not exist
        # with `log_divice_placement=True` we can see all the operations and tensors are mapped to which device
        session_config = tf.ConfigProto(allow_soft_placement=True, 
            log_device_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
    
    model_dir = os.path.join(FLAGS.output_dir, "model_dir", "cnn"
                '-'.join(
                    ["nblock_" + str(FLAGS.resnet_layers), "filtersz_" + str(FLAGS.filters), 
                    "input_" + str(FLAGS.input_size), "output_" + str(FLAGS.output_size),
                    "bs_" + str(FLAGS.batch_size)]))

    if FLAGS.platform == "npu":
        run_config = NPURunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=None,
            # precision_mode="allow_mix_precision"
        )
    else:
        # allow mixed precision
        # session_config.graph_options.rewrite_options.auto_mixed_precision=1
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
        with tf.contrib.tfprof.ProfileContext(model_dir, dump_steps=[10]) as pctx:
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

if __name__ == '__main__':
    tf.app.run()