'''
Fully-connected models.

@author Weizheng Lu
'''

import tensorflow as tf
import time
import numpy as np
import os
import resnet_cifar10

# whether use NPU or not
tf.flags.DEFINE_string("platform", "gpu", "which computing platform we are using, GPU/NPU")
tf.flags.DEFINE_string("output_dir", "./", "All the output data should be written into this folder.")
tf.flags.DEFINE_string("train_dir", "./", "Training data should be written into this folder.")
tf.flags.DEFINE_integer("iterations", 100,
                        "Number of iterations per training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (accelerator chips).")
tf.flags.DEFINE_string("machine", 'npu/2x2', "machine")

tf.flags.DEFINE_integer("train_steps", 100, "Total number of steps.")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training.")
tf.flags.DEFINE_string("mode", "train", "train or inference.")
tf.flags.DEFINE_integer("warmup_steps", 300, "warmup steps")
tf.flags.DEFINE_integer("input_size", 32, "input_size")
tf.flags.DEFINE_integer("output_size", 10, "output_size")
tf.flags.DEFINE_integer("filters", 16, "number of filters or channels of the image")
tf.flags.DEFINE_string("resnet_layers", "1,1,1,1", "residual blocks in each group")
tf.flags.DEFINE_string("block_fn", "residual", "choose from residual and bottleneck")

tf.flags.DEFINE_string("optimizer", "rms", "Choose among rms, sgd, and momentum.")
tf.flags.DEFINE_string("data_type", "float32", "")

FLAGS = tf.flags.FLAGS

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

def get_input_fn(filenames, batch_size=128, shuffle=False):
    def _parser(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'raw_image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['raw_image'], tf.uint8)
        image.set_shape([3 * 32 * 32])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        return image, tf.one_hot(label, depth=10)

    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.repeat(None)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

def model_fn(features, labels, mode, params):
    output_size = 10
    print("------------- before ----------------", features.get_shape())
    net = features

    if FLAGS.data_type == 'float32':
        network = resnet_cifar10.resnet_v1(
            resnet_layers,
            block_fn,
            num_classes=output_size,
            data_format='channels_last',
            filters=filters)

        logits = network(inputs=features, is_training=True)
        
    # onehot_labels=tf.one_hot(labels, output_size)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=labels, logits=logits)
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    scores = tf.nn.softmax(logits, name='softmax_tensor')
    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predicted_logit)
    
    train_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(labels, axis=1, output_type=tf.int32), predicted_logit), tf.float32)
        )

    learning_rate = tf.train.exponential_decay(
        0.1, tf.train.get_global_step(), 25000, 0.97)
    
    eval_metric = { 'test_accuracy': accuracy }
    
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
            tensors={"loss" : loss, "train_acc": train_accuracy},
            every_n_iter=10
        )
    # train
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=[train_hook],
        eval_metric_ops=eval_metric)

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


    if FLAGS.platform == "npu":
        run_config = NPURunConfig(
            # model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=10,
            # precision_mode="allow_mix_precision"
        )
    else:
        # allow mixed precision
        # session_config.graph_options.rewrite_options.auto_mixed_precision=1
        run_config = tf.estimator.RunConfig(
            # model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=10
        )

    if FLAGS.platform == "npu":
        estimator = NPUEstimator(
            model_fn=model_fn,
            config=run_config
        )
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config
        )

    if FLAGS.mode == 'train':
        train_filenames = [os.path.join(FLAGS.train_dir,'data_batch_%d.tfrecord' %i) for i in range(1,6)]
        train_input_fn = get_input_fn(train_filenames, batch_size=FLAGS.batch_size)
        eval_input_fn = get_input_fn(os.path.join(FLAGS.train_dir, 'eval.tfrecord'), batch_size=FLAGS.batch_size)
        # estimator.train(input_fn=train_input_fn, max_steps=10000)
        
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=10000)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        start = time.time()
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        
    # else:
        # tf.logging.info('Running estimator.predict()')
        # estimator.train(input_fn=get_input_fn(input_size, output_size), max_steps=1)
        # p = estimator.evaluate(input_fn=get_input_fn(input_size, output_size), steps=FLAGS.warmup_steps)
        # start = time.time()
        # p = estimator.evaluate(input_fn=get_input_fn(input_size, output_size), steps=FLAGS.train_steps)

    total_time = time.time() - start
    example_per_sec = batch_size * train_steps / total_time
    global_step_per_sec = train_steps / total_time
    print("Total time: " + str(total_time))
    #tf.logging.info("global_step/sec: %s" % global_step_per_sec)
    #tf.logging.info("examples/sec: %s" % example_per_sec)

if __name__ == '__main__':
    tf.app.run()