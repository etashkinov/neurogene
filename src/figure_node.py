import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import tensorflow as tf

from src import data_loader

activations = ('relu', 'tanh')
optimizers = ('sgd', 'adam')
losses = ('categorical_crossentropy', 'whatever')
regulizers = ('L1', 'L2')


def create_network(params):
    name = "net_" + str(params) \
        .replace('(', '') \
        .replace(')', '') \
        .replace(',', '') \
        .replace(' ', '') \
        .replace('.', '')

    print('Create network:' + name)
    network = input_data(shape=[None, 28, 28, 1], name='input', dtype=tf.float32)
    layer_n = 0
    for p in params[:-1]:
        layer_n = layer_n + 1
        layer_type = p[0]
        shape = p[1]
        activation = activations[p[2]]
        regulizer = regulizers[p[3]]
        drpt = p[4]
        fltr = p[5]
        max_pool = p[6]

        if layer_type == 0:
            layer_name = 'fc_' + str(layer_n)
            print('Create layer {} with shape: {}, activation: {}, dropout: {}'
                  .format(layer_name, shape, activation, drpt))
            network = fully_connected(network, shape,
                                      activation=activation,
                                      name=layer_name)
            network = dropout(network, drpt)
        elif layer_type == 1:
            layer_name = 'conv_' + str(layer_n)
            print('Create layer {} with shape: {}, activation: {}, regularizer: {}, filter: {}, max_pool: {}'
                  .format(layer_name, shape, activation, regulizer, fltr, max_pool))
            network = conv_2d(network, shape, fltr,
                              activation=activation,
                              regularizer=regulizer,
                              name=layer_name)
            network = max_pool_2d(network, p[6])
            network = local_response_normalization(network)

    network = fully_connected(network, 3, activation='softmax', name='prob')

    optimizer = optimizers[params[-1][0]]
    learning_rate = params[-1][1]
    loss = losses[params[-1][2]]
    print('Add regression {} at {} with {}'.format(optimizer, learning_rate, loss))
    network = regression(network,
                         optimizer=optimizer,
                         learning_rate=learning_rate,
                         loss=loss,
                         name='target')

    return network, name


def fit(train, test, epoch, params):
    train_x = np.array([t.encoding for t in train])
    train_y = np.array([t.one_hot for t in train])

    test_x = np.array([t.encoding for t in test])
    test_y = np.array([t.one_hot for t in test])

    graph = tf.Graph()
    with graph.as_default():
        network, name = create_network(params)
        model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='board')
        model.fit({'input': train_x}, {'target': train_y}, n_epoch=epoch,
                  validation_set=({'input': test_x}, {'target': test_y}),
                  snapshot_step=100, show_metric=True, run_id=name)

        model.save(name)

        return model, name


train = data_loader.get_data('../figures')
test = data_loader.get_data('../objects')

prediction = fit(train, test, 100, (
    (1, 32, 0, 1, 0, 3, 2),
    (1, 64, 0, 1, 0, 3, 2),
    (0, 128, 1, 0, 0.8, 0, 0),
    (0, 256, 1, 0, 0.8, 0, 0),
    (1, 0.01, 0)
))
print(prediction)
