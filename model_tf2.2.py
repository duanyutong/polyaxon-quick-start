''' All store manager commands
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
import threading
import argparse
from glob import glob

import tensorflow as tf

from polyaxon_client.tracking import Experiment, get_log_level, get_data_paths


output_dir = '/outputs'


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str
    )
    parser.add_argument(
        '--dropout',
        default=0.25,
        type=float
    )
    parser.add_argument(
        '--activation',
        default='relu',  # sigmoid, tanh, linear etc.
        type=str
    )
    parser.add_argument(
        '--num_epochs',
        default=1,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        default=100,
        type=int
    )
    return parser

def get_logger():
    # calling getLogger without a name or with __name__ will cause multiple
    # threads in the same namespace to compete for a file handle
    # don't re-use any logger from any other threads by hashing the input args
    name = 'TID-{} {}.{}'.format(threading.get_ident(), __file__, __name__)
    logger = logging.getLogger(name)
    lvl = logging.getLevelName(get_log_level())
    logger.setLevel(lvl)  # follow config lvl
    sh = logging.StreamHandler()  # get stream handler
    formatter = logging.Formatter(  # log format for each line
        fmt=('%(asctime)s [%(levelname)-8s] '
             '%(name)s.%(funcName)s+%(lineno)s: %(message)s'),
        datefmt='%Y-%m-%dT%H:%M:%S%z')
    sh.setFormatter(formatter)
    logger.addHandler(sh)  # add stream handler
    logging.getLogger("tensorflow").setLevel(lvl)  # also set tf to same level
    return logger


def create_model(activation, dropout, optimizer):
    # Build the tf.keras.Sequential model by stacking layers.
    # Choose an optimizer and loss function for training:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(10)])
    # The losses.SparseCategoricalCrossentropy loss takes a vector of logits
    # and a True index and returns a scalar loss for each example.
    # This loss is equal to the negative log probability of the true class:
    # It is zero if the model is sure of the correct class.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model


def save_model(model):
    model.save(os.path.join(output_dir, 'model.h5'))
    model.save(os.path.join(output_dir, 'weights.h5'))
    model.to_yaml(os.path.join(output_dir, 'network.yaml'))


def upload_outputs(experiment):
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


class MyCustomCallback(tf.keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        print(f'Training batch {batch} begins')

    def on_train_batch_end(self, batch, logs=None):
        print(f'Training batch {batch} ends')

    def on_test_batch_begin(self, batch, logs=None):
        print(f'Test batch {batch} begins')

    def on_test_batch_end(self, batch, logs=None):
        print(f'Test batch {batch} ends')

    def on_test__end(self, logs=None):
        print(f'Test ends, logs:\n{logs}')


if __name__ == '__main__':

    args = get_argparser().parse_args()
    logger = get_logger()
    logger.debug('Data paths available: {}'.format(get_data_paths()))
    experiment = Experiment()
    mnist = tf.keras.datasets.mnist  # Load and prepare the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = create_model(args.activation, args.dropout, args.optimizer)
    model.fit(x_train, y_train,
              batch_size=args.batch_size, epochs=args.num_epochs)
            #   callbacks=[MyCustomCallback])
    # The image classifier gets to ~98% after 5 epochs
    loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
    save_model(model)
    upload_outputs(experiment)


    # for i in range(num_epochs):

    #     metrics = estimator.evaluate(input_fn)

    #     logger.info("Testing metrics: {}".format(metrics))
    #     experiment.log_metrics(loss=metrics['loss'],
    #                             accuracy=metrics['accuracy'],
    #                         precision=metrics['precision'])
    logger.critical('This is a sample critical log message at the end.')
