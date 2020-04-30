'''
cluster instructions

* multiple data volumes are mounted including "/data" and s3 bucketes,
  use tracking.get_data_paths to see all data paths

  * access images and mpacks (indexed by future DB) by
    store.download_file(filename, local_path),  maybe will write a wrapper

* only one output storage is supported, and we have s3 as default.

  * always store output locally (in container) in "/outputs"
    with your preferred directory structure, and upload to s3 when finished
      experiment.log_artifact(file_path)
      experiment.log_artifacts(dir_path)
    which are equivalent to deprecated
      experiment.outputs_store.upload_file(file_path)
      experiment.outputs_store.upload_dir(dir_path)
  * no need to call get_output_path



from polystores.stores.manager import StoreManager
store = StoreManager(path=data_path)

store.delete(path)
store.ls(path)
store.upload_file(filename)
store.upload_dir(dirname)
store.download_file(filename, local_path)
store.download_dir(dirname, local_path)
'''


import os
import logging
import argparse
import tensorflow as tf
from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths
from tensorflow.examples.tutorials.mnist import input_data
from glob import glob



def get_logger():
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.getLevelName(get_log_level()))  # follow config lvl
    sh = logging.StreamHandler()  # get stream handler
    formatter = logging.Formatter(  # log format for each line
        fmt='%(asctime)s %(name)s [%(levelname)-8s]: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S%z')
    sh.setFormatter(formatter)
    logger.addHandler(sh)  # add stream handler
    return logger

def set_logging(log_level=None):
    if log_level == 'INFO':
        log_level = tf.logging.INFO
    elif log_level == 'DEBUG':
        log_level = tf.logging.DEBUG
    elif log_level == 'WARN':
        log_level = tf.logging.WARN
    else:
        log_level = 'INFO'
    tf.logging.set_verbosity(log_level)


set_logging(get_log_level())

def get_model_fn(learning_rate, dropout, activation):
    """Create a `model_fn` compatible with tensorflow estimator based on hyperparams."""

    def get_network(x_dict, is_training):
        with tf.variable_scope('network'):
            x = x_dict['images']
            x = tf.reshape(x, shape=[-1, 28, 28, 1])
            conv1 = tf.layers.conv2d(x, 32, 5, activation=activation)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=activation)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            fc1 = tf.contrib.layers.flatten(conv2)
            fc1 = tf.layers.dense(fc1, 1024)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
            out = tf.layers.dense(fc1, 10)
        return out

    def model_fn(features, labels, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        results = get_network(features, is_training=is_training)

        predictions = tf.argmax(results, axis=1)

        # Return prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Define loss
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=results, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluation metrics
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        precision = tf.metrics.precision(labels=labels, predictions=predictions)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': accuracy, 'precision': precision})

    return model_fn


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int
    )
    parser.add_argument(
        '--num_steps',
        default=800,
        type=int
    )
    parser.add_argument(
        '--num_iterations',
        default=1,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--dropout',
        default=0.25,
        type=float
    )
    parser.add_argument(
        '--num_epochs',
        default=1,
        type=int
    )
    parser.add_argument(
        '--activation',
        default='relu',
        type=str
    )
    parser.add_argument(
        '--distributed',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--logger_type',
        default='logger',
        type=str
    )

    args = parser.parse_args()
    arguments = args.__dict__

    logger = get_logger()
    logger.info('Logger initialised')
    logger.debug('data paths available: {}'.format(get_data_paths()))
    data_dir = get_data_paths()['local'] + "/mnist"  # TF doesn't support s3
    logger.warning('caching temp data to default local storage: {}'.format(
        data_dir))
    mnist = input_data.read_data_sets(data_dir, one_hot=False)
    output_dir = '/outputs'  # temporary

    batch_size = arguments.pop('batch_size')
    num_steps = arguments.pop('num_steps')
    learning_rate = arguments.pop('learning_rate')
    dropout = arguments.pop('dropout')
    num_epochs = arguments.pop('num_epochs')
    activation = arguments.pop('activation')
    distributed = arguments.pop('distributed')
    num_iterations = arguments.pop('num_iterations')
    if activation == 'relu':
        activation = tf.nn.relu
    elif activation == 'sigmoid':
        activation = tf.nn.sigmoid
    elif activation == 'linear':
        activation = None

    experiment = Experiment()
    if distributed:
        # Check if we need to export TF_CLUSTER
        experiment.get_tf_config()

    estimator = tf.estimator.Estimator(
        get_model_fn(learning_rate=learning_rate, dropout=dropout, activation=activation),
        model_dir=output_dir)

    # Train the Model
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images},
        y=mnist.train.labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True)

    for i in range(num_iterations):
        estimator.train(input_fn, steps=num_steps)

        # Evaluate the Model
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': mnist.test.images},
            y=mnist.test.labels,
            batch_size=batch_size,
            shuffle=False)

        metrics = estimator.evaluate(input_fn)

        logger.info("Testing metrics: {}".format(metrics))
        experiment.log_metrics(loss=metrics['loss'],
                               accuracy=metrics['accuracy'],
                               precision=metrics['precision'])
    for path in glob(os.path.join(output_dir, '*')):
        if os.path.isfile(path):
            logger.info('Logging artifact file: {}'.format(path))
            experiment.log_artifact(path)
        elif os.path.isdir(path):
            logger.info('Logging artifact dir: {}'.format(path))
            experiment.log_artifacts(path)
        else:
            logger.error('Path skipped for being neither a file nor '
                         'a directory: {}'.format(path))
    logger.critical('This is a sample critical log message at the end.')
