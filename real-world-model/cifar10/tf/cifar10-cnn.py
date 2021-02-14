from argparse import ArgumentParser
import os
import tensorflow as tf

tf.flags.DEFINE_string("platform", "gpu", "which computing platform we are using, GPU/NPU")
tf.flags.DEFINE_string("output_dir", "./", "All the output data should be written into this folder.")
tf.flags.DEFINE_integer("iterations", 100,
                        "Number of iterations per training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (accelerator chips).")

tf.flags.DEFINE_integer("epochs", 100, "Total number of epochs.")
tf.flags.DEFINE_integer("train_batch_size", 128,
                        "Mini-batch size for the training.")
tf.flags.DEFINE_integer("eval_batch_size", 128,
                        "Mini-batch size for the evaluating.")
tf.flags.DEFINE_string("mode", "train", "train or inference.")
tf.flags.DEFINE_float("dropout_rate", 0.4, "dropout rate")
tf.flags.DEFINE_float("learning_rate", 0.0007, "learning rate")
tf.flags.DEFINE_string("train_dir", "./train", "train data directory.")
tf.flags.DEFINE_string("eval_dir", "./eval", "evaluate data directory.")
tf.flags.DEFINE_string("optimizer", "rms", "Choose among rms, sgd, and momentum.")
tf.flags.DEFINE_string("data_type", "float32", "float32 or mix precision")

FLAGS = tf.flags.FLAGS
model_dir = FLAGS.output_dir

def cnn_model(features, mode, params):

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    #print("------------- before ----------------", features.get_shape())
    with tf.name_scope('Input'):
        # Input Layer
        input_layer = tf.reshape(features, [-1, 32, 32, 3], name='input_reshape')
        tf.summary.image('input', input_layer)
    #print("------------- after -----------------", input_layer.get_shape())

    with tf.name_scope('Conv_1'):
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=(5, 5),
          padding='same',
          activation=tf.nn.relu,
          trainable=is_training,
          data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv1', conv1)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, padding='same')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.name_scope('Conv_2'):
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=norm1,
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training,
            data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv2', conv2)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, padding='same')
        norm2 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    with tf.name_scope('Conv_3'):
        # Convolutional Layer #3 and Pooling Layer #3
        conv3 = tf.layers.conv2d(
            inputs=norm2,
            filters=96,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training,
            data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv3', conv3)

        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2, padding='same')
        norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    with tf.name_scope('Conv_4'):
        # Convolutional Layer #4 and Pooling Layer #4
        conv4 = tf.layers.conv2d(
            inputs=norm3,
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation=tf.nn.relu,
            trainable=is_training,
            data_format='channels_last')
        tf.summary.histogram('Convolution_layers/conv4', conv4)

        norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
        pool4 = tf.layers.max_pooling2d(inputs=norm4, pool_size=(2, 2), strides=1, padding='same')


    with tf.name_scope('Dense_Dropout'):
        # Dense Layer
        pool_flat = tf.contrib.layers.flatten(pool4)
        dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu, trainable=is_training)
        dropout = tf.layers.dropout(inputs=dense, rate=FLAGS.dropout_rate, training=is_training)
        tf.summary.histogram('fully_connected_layers/dropout', dropout)


    with tf.name_scope('Predictions'):
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10, trainable=is_training)
        return logits

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""

    logits = cnn_model(features, mode, params)
    predicted_logit = tf.argmax(input=logits, axis=1, output_type=tf.int32)
    scores = tf.nn.softmax(logits, name='softmax_tensor')

    # Generate Predictions
    predictions = {
      'classes': predicted_logit,
      'probabilities': scores
    }

    export_outputs = {
        'prediction': tf.estimator.export.ClassificationOutput(
            scores=scores,
            classes=tf.cast(predicted_logit, tf.string))
    }

    # For PREDICTION mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # For TRAIN and EVAL modes
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predicted_logit)
    train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1, output_type=tf.int32), predicted_logit), tf.float32))

    eval_metric = { 'test_accuracy': accuracy }

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
            
            )
    else:
        train_op = None
    tf.summary.scalar('train_accuracy', train_accuracy)
    summary_op = tf.summary.merge_all()
    summary_train_hook = tf.train.SummarySaverHook(
        save_steps=10,
        output_dir=model_dir,
        summary_op=summary_op)
    # EstimatorSpec fully defines the model to be run by an Estimator.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric, # A dict of name/value pairs specifying the metrics that will be calculated when the model runs in EVAL mode.
        predictions=predictions,
        export_outputs=export_outputs,
        training_hooks=[summary_train_hook])

def data_input_fn(filenames, batch_size=128, shuffle=False):
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

def main(unused_argv):
    del unused_argv
    tf.logging.set_verbosity(tf.logging.INFO)
    print('\nTensorFlow version: ' + str(tf.__version__))

    if FLAGS.platform == "npu":
        session_config = tf.ConfigProto()
    elif FLAGS.platform == "gpu":
        # with `allow_soft_placement=True` TensorFlow will automatically help us choose a device in case the specific one does not exist
        # with `log_divice_placement=True` we can see all the operations and tensors are mapped to which device
        session_config = tf.ConfigProto(
            allow_soft_placement=True, 
            log_device_placement=True, 
            gpu_options=tf.GPUOptions(allow_growth=True)
        )

    if FLAGS.platform == "npu":
        run_config = NPURunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_secs=10,
            save_summary_steps=10,
            precision_mode="allow_mix_precision"
        )
    elif FLAGS.platform == "gpu":
        # mixed precision
        if FLAGS.data_type == "mix":
            session_config.graph_options.rewrite_options.auto_mixed_precision=1
        run_config = tf.estimator.RunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_steps=10,
            save_summary_steps=10
        )

    if FLAGS.platform == "npu":
        estimator = NPUEstimator(
            model_fn=cnn_model_fn,
            model_dir=model_dir,
            config=run_config
        )
    else:
        estimator = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=model_dir,
            config=run_config
        )

    train_filenames = [os.path.join(FLAGS.train_dir,'data_batch_%d.tfrecord' %i) for i in range(1,6)]
    train_input_fn = data_input_fn(train_filenames, batch_size=FLAGS.train_batch_size)
    eval_input_fn = data_input_fn(os.path.join(FLAGS.eval_dir, 'eval.tfrecord'), batch_size=FLAGS.eval_batch_size)

    estimator.train(input_fn=train_input_fn, steps=FLAGS.train_batch_size * FLAGS.epochs)

if __name__ == '__main__':
    tf.app.run()
