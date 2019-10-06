import pickle
import numpy as np
import nengo_dl
import nengo
import tensorflow as tf
def toOneHot(labels, classNum):
    oneHotLabels = np.zeros(shape=[labels.shape[0], classNum])
    for i in range(labels.shape[0]):
        oneHotLabels[i][labels[i]] = 1
    return oneHotLabels

train_data, validation_data, test_data = pickle.load(open('../datasets/mnist.pkl', 'rb'), encoding="latin1")
x_train, y_train = train_data[0], train_data[1]
x_test, y_test = test_data[0], test_data[1]


# one-hot
y_train = toOneHot(y_train, 10)
y_test = toOneHot(y_test, 10)

with nengo.Network() as net:
    # set some default parameters for the neurons that will make
    # the training progress more smoothly
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    neuron_type = nengo.LIF(amplitude=0.01)

    # we'll make all the nengo objects in the network
    # non-trainable. we could train them if we wanted, but they don't
    # add any representational power. note that this doesn't affect
    # the internal components of tensornodes, which will always be
    # trainable or non-trainable depending on the code written in
    # the tensornode.
    nengo_dl.configure_settings(trainable=False)

    # the input node that will be used to feed in input images
    inp = nengo.Node(output=[0] * 28 * 28) # 输出是784

    # tensor_layer层可以自动转换输入的shape，转换时不会转换第0维（batch）
    # add the first convolutional layer
    x = nengo_dl.tensor_layer(
        inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
        kernel_size=3)

    # apply the neural nonlinearity
    x = nengo_dl.tensor_layer(x, neuron_type)

    # add another convolutional layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.conv2d, shape_in=(26, 26, 32),
        filters=64, kernel_size=3)
    x = nengo_dl.tensor_layer(x, neuron_type)

    # add a pooling layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.average_pooling2d, shape_in=(24, 24, 64),
        pool_size=2, strides=2)

    # another convolutional layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.conv2d, shape_in=(12, 12, 64),
        filters=128, kernel_size=3)
    x = nengo_dl.tensor_layer(x, neuron_type)

    # another pooling layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.average_pooling2d, shape_in=(10, 10, 128),
        pool_size=2, strides=2)

    # linear readout
    x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

    # we'll create two different output probes, one with a filter
    # (for when we're simulating the network over time and
    # accumulating spikes), and one without (for when we're
    # training the network using a rate-based approximation)
    out_p = nengo.Probe(x)
    out_p_filt = nengo.Probe(x, synapse=0.1)

minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size, device='/device:XLA_GPU:0')

# 数据需要在时间上也有一个维度
# 原来是(50000, 784)
# 转换成(50000, 1, 784)
train_data = {inp: x_train[:, None, :],
              out_p: y_train[:, None, :]}
n_steps = 10
test_data = {
    inp: np.tile(x_test[:minibatch_size*2, None, :],
                 (1, n_steps, 1)),
    out_p_filt: np.tile(y_test[:minibatch_size*2, None, :],
                        (1, n_steps, 1))}

def objective(outputs, targets):
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets)

opt = tf.train.AdamOptimizer(learning_rate=0.001)
def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))

print("error before training: %.2f%%" % sim.loss(
    test_data, {out_p_filt: classification_error}))

sim.train(train_data, opt, objective={out_p: objective}, n_epochs=1)

print("error after training: %.2f%%" % sim.loss(
    test_data, {out_p_filt: classification_error}))

sim.close()
exit(0)