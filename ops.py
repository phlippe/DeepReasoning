import os
import math

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import inspect_checkpoint as chkp


IS_OLD_TENSORFLOW = (tf.__version__[0] == '0')
WEIGHT_DECAY_FACTOR = tf.constant(2e-6, name="WeightDecayFactor")


def get_vocab_variable(name, shape, scale_factor=1e-3):
    def vocab_regularization(weights):
        with tf.name_scope("Regularization"):
            return scale_factor*tf.reduce_mean(tf.pow(weights, 2) / 2.0)
    vocab = tf.get_variable(name,
                            initializer=tf.random_uniform(shape=shape,
                                                          minval=-1.0,
                                                          maxval=1.0,
                                                          dtype=tf.float32),
                            regularizer=vocab_regularization,
                            dtype=tf.float32)
    tf.summary.scalar(name="Voc_"+name+"_Max", tensor=tf.reduce_max(input_tensor=vocab))
    tf.summary.scalar(name="Voc_"+name+"_Min", tensor=tf.reduce_min(input_tensor=vocab))
    tf.summary.scalar(name="Voc_"+name+"_Var", tensor=tf.reduce_mean(input_tensor=(tf.pow(vocab, 2) / 2.0)))
    return vocab


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=1, d_w=1,
           name="conv2d", reuse=False, padding='SAME', relu=False, use_batch_norm=False):
    """
    Builds up a convolution layer on input including weights and biases.
    :param input_: Input to convolution with shape NHWC
    :param output_dim: Channel size of output
    :param k_h: Kernel height
    :param k_w: Kernel width
    :param d_h: Stride size along height dimension
    :param d_w: Stride size along width dimension
    :param name: Name of layer
    :param reuse: If variables should be reused
    :param padding: Padding of convolution layer
    :return: Output of convolution with shape NHWC (channel size = output_dim)
    """
    global WEIGHT_DECAY_FACTOR
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY_FACTOR))
        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY_FACTOR))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        if relu:
            conv = tf.nn.relu(conv)

        if use_batch_norm:
            conv = tf.layers.batch_normalization(conv)

        return conv


def dilated_conv2d(input_, output_dim,
                   k_h=3, k_w=3, dilation_rate=1,
                   name="dil_conv2d", reuse=False, padding='SAME'):
    """
    Builds up a dilated convolution layer on input including weights and biases.
    Args:
      input_ - Input to the layer
      output_dim - Number of output channels
      k_h - Kernel height
      k_w - Kernel width
      dilation_rate - Dilation rate (for details see Tensorflow atrous_conv2d)
      name - Name of variable scope
      reuse - If variables should be reused
      padding - Padding of convolution layer
    """
    global WEIGHT_DECAY_FACTOR
    with tf.variable_scope(name, reuse=reuse):
        input_shape = input_.get_shape().as_list()
        w = tf.get_variable('w', [k_h, k_w, input_shape[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY_FACTOR))
        conv = tf.nn.atrous_conv2d(
            input_, w, rate=dilation_rate, padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY_FACTOR))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), shape=[input_shape[0], input_shape[1], -1, input_shape[3]])
        return conv


def conv1d(input_, output_dim, kernel_size=3, d_h=1, d_w=1,
           name="conv1d", reuse=False, padding='SAME', relu=False, use_batch_norm=False):
    """
    Builds up a one dimensional convolution layer on input including weights and biases.
    :param input_: Input to convolution with shape NHWC
    :param output_dim: Channel size of output
    :param kernel_size: Kernel size
    :param d_h: Stride size along height dimension
    :param d_w: Stride size along width dimension
    :param name: Name of layer
    :param reuse: If variables should be reused
    :param padding: Padding of convolution layer
    :param relu: If activation function "ReLU" should be used or not
    :return: Output of convolution with shape NHWC (channel size = output_dim)
    """
    input_shape = input_.get_shape()
    if len(input_shape) == 3:
        input_ = tf.reshape(input_, shape=(input_shape[0], 1, input_shape[1], input_shape[2]))
    return conv2d(input_=input_, output_dim=output_dim, k_h=1, k_w=kernel_size, d_h=d_h, d_w=d_w,
                  name=name, reuse=reuse, padding=padding, relu=relu, use_batch_norm=use_batch_norm)


def dilated_conv1d(input_, output_dim, k_w=3, dilation_rate=1,
                   name="dil_conv1d", reuse=False, padding='SAME'):
    """
    Builds up a dilated convolution layer on input including weights and biases.
    Args:
      input_ - Input to the layer, shape BWC or B1WC (B - Batch, W - Width, C - Channels)
      output_dim - Number of output channels
      k_w - Kernel size (One dimensional)
      dilation_rate - Dilation rate (for details see Tensorflow atrous_conv2d)
      name - Name of variable scope
      reuse - If variables should be reused
      padding - Padding of convolution layer

    :return: Convolved input with shape [batch_size, 1, width, output_dim]
    """
    input_shape = input_.get_shape().as_list()
    if len(input_shape) == 3:
        input_ = tf.reshape(input_, shape=(input_shape[0], 1, input_shape[1], input_shape[2]))
    return dilated_conv2d(input_=input_, output_dim=output_dim, k_h=1, k_w=k_w, dilation_rate=dilation_rate,
                          name=name, reuse=reuse, padding=padding)


def fast_dilated_conv1d(input_, output_dim, k_w=3, dilation_rate=1, name="fast_dilconv1d", reuse=False):
    with tf.name_scope(name):
        if dilation_rate != 1:
            input_shape = input_.get_shape().as_list()
            if len(input_shape) == 3:
                input_ = tf.expand_dims(input_, axis=1)

            # Starting with shape [batch, 1, width, channels]
            input_shape = tf.shape(input_)
            input_shape_list = input_.get_shape().as_list()
            # First fill up elements so that width is divisible by dilation
            pad_elements = dilation_rate - 1 - tf.mod((input_shape[2] + dilation_rate - 1), dilation_rate)
            input_ = tf.pad(input_, [[0, 0], [0, 0], [0, pad_elements], [0, 0]])
            # Convert shape to [batch, width/dilation, dilation, channels]
            # => over height dimension we have every dilation^th element
            reshaped_input = tf.reshape(input_, shape=[input_shape_list[0],
                                                       tf.cast(tf.divide(tf.shape(input_)[2], int(dilation_rate)), tf.int32),
                                                       dilation_rate,
                                                       input_shape_list[3]])
            # Run a standard 1d-convolution over height(!)-dimension
            out = conv2d(input_=reshaped_input, output_dim=output_dim, k_h=k_w, k_w=1, d_h=1, d_w=1, padding='SAME',
                         reuse=reuse, relu=False, use_batch_norm=False, name=name)
            # Reshape output back to original input shape
            reshaped_output = tf.reshape(out, shape=[input_shape_list[0],
                                                     1,
                                                     tf.shape(input_)[2],
                                                     input_shape_list[3]])
            return reshaped_output[:, :, :input_shape[2], :]   # Cut out added elements
        else:
            return conv1d(input_=input_, output_dim=output_dim, kernel_size=k_w, d_h=1, d_w=1, reuse=reuse,
                          relu=False, use_batch_norm=False, name=name)


def wavenet_layer(input_, kernel_size=3, dilation_rate=1, name="WaveNet_Layer", reuse=False):
    """
    WaveNet Layer as defined in "Deep Network Guided Proof Search" (Sarah Loss et. al):

    Ld(x) = x + tanh(Cd(x))sigmoid(Cd'(x))

    with Cd, Cd' as dilated convolutions.

    :param input_: Input to layer (x in equation), shape BWC or B1WC (B - Batch, W - Width, C - Channels)
    :param kernel_size: Size of one-dimensional kernel, which is applied in the dilated convolution
    :param dilation_rate: Dilation layer used for dilated_conv1d
    :param name: Name of layer block
    :param reuse: If variables should be reused or not
    :return: Output of the combined layer with shape B1WC (B - Batch, W - Width, C - Channels) as input
    """
    with tf.variable_scope(name):
        input_shape = input_.get_shape().as_list()
        channel_size = input_shape[-1]
        filter_layer = tf.nn.tanh(
            dilated_conv1d(input_=input_, output_dim=channel_size, k_w=kernel_size, dilation_rate=dilation_rate,
                           name=name + "_filter", reuse=reuse))
        gate_layer = tf.nn.sigmoid(
            dilated_conv1d(input_=input_, output_dim=channel_size, k_w=kernel_size, dilation_rate=dilation_rate,
                           name=name + "_gate", reuse=reuse))
        layer_d = filter_layer * gate_layer + input_
        return layer_d


def hierarchical_wavenet_block(input_tensor, block_number, layer_number, kernel_size=3, dropout_rate=0.2, reuse=False,
                               name="Hierarchical_WaveNet_Block"):
    with tf.variable_scope(name):
        block_tensor = None
        for block_index in range(block_number):
            with tf.variable_scope("Block_" + str(block_index)):
                """
                B(x) = x + (L_{64} * L_{32} * ... * L_{1})(D_{f}(x,p))
                """
                if block_index == 0:
                    block_tensor = dropout(input_tensor, dropout_rate)
                else:
                    block_tensor = dropout(block_tensor, dropout_rate)
                layer_tensor = block_tensor

                for layer_index in range(layer_number):
                    layer_tensor = wavenet_layer(input_=layer_tensor, kernel_size=kernel_size,
                                                 dilation_rate=2 ** layer_index,
                                                 name="WaveNetLayer_" + str(layer_index), reuse=reuse)
                    print("(Block "+str(block_index)+", Layer "+str(layer_index)+") -> Layer tensor shape: "+str(layer_tensor.get_shape().as_list()))
                block_tensor = block_tensor + layer_tensor
        return block_tensor


def dilated_dense_block(input_tensor, layer_number, kernel_size=3, channel_size=-1, end_channels=-1, reuse=False,
                        dropout_rate=0.5, training=False, name="DilatedDenseBlock"):
    if channel_size <= 0:
        channel_size = input_tensor.get_shape().as_list()[-1]
    if end_channels <= 0:
        end_channels = channel_size
    all_layers = [input_tensor]
    output_tensor = input_tensor
    with tf.variable_scope(name):
        for layer_index in range(layer_number):
            with tf.name_scope("DilConv"+str(layer_index)):
                layer_out = tf.nn.relu(
                                  fast_dilated_conv1d(input_=output_tensor, output_dim=channel_size, k_w=kernel_size,
                                                      dilation_rate=2 ** layer_index, name="DilConv"+str(layer_index),
                                                      reuse=reuse),
                                  name="RELU_"+str(layer_index)
                            )
                layer_out = dropout(layer_out, dropout_rate, training)
            all_layers.append(layer_out)
            with tf.name_scope("FeatureReduction_"+str(layer_index)):
                output_tensor = tf.nn.relu(conv1d(input_=tf.concat(values=all_layers, axis=3, name="FeatureConcat"+str(layer_index)), kernel_size=1,
                                                  output_dim=channel_size if layer_index + 1 < layer_number else end_channels,
                                                  name="FeatureReduction_"+str(layer_index), reuse=reuse), name="RELU_"+str(layer_index))
                output_tensor = dropout(output_tensor, dropout_rate, training)
    return output_tensor


def dropout(input_, p, training=False):
    """
    Dropout layer.
    :param input_: Input to dropout
    :param p: Probability with which a node is set to zero
    :param training: Whether the network is training or testing
    :return: Input with applied dropout
    """
    return tf.layers.dropout(inputs=input_, rate=p, training=training)


def fully_connected(input_, outputs, activation_fn=tf.nn.relu, reuse=False, name="FC_Layer", use_batch_norm=False):
    """
    Fully connected layer
    :param input_:
    :param outputs:
    :param activation_fn:
    :param reuse:
    :param name:
    :return:
    """
    input_shape = input_.get_shape()
    if len(input_shape) == 2:
        input_ = tf.reshape(tensor=input_, shape=[input_shape[0], 1, 1, input_shape[1]])

    with tf.variable_scope(name):
        # Earlier: fc = tf.contrib.layers.fully_connected(input_, outputs, activation_fn, reuse=reuse)
        # Now using 1x1 convolution for possible height dimension
        fc = activation_fn(conv1d(input_=input_, output_dim=outputs, kernel_size=1, relu=False, reuse=reuse, name=name,
                                  use_batch_norm=False))
        if use_batch_norm:
            fc = tf.layers.batch_normalization(fc)
        return fc


def save_model(saver, sess, checkpoint_dir, step, model_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)


def load_model(saver, sess, checkpoint_dir, model_name=None):
    """
    Function for loading the model. Copied from https://github.com/rubenvillegas/iclr2017mcnet/blob/master/src/mcnet.py
    MIT License - downloaded 07/01/2017
    """
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        if model_name is None:
            model_name = ckpt_name
        saver.restore(sess, os.path.join(checkpoint_dir, model_name))
        return True
    else:
        return False


def freeze_graph(model_folder, output_node_names, file_name="frozen_model.pb"):
    """
    Freezes a graph that has saved checkpoints in the given folder.
    :param model_folder:
    :param output_node_names: Before exporting our graph, we need to precise what is our output node. This is how TF
                              decides what part of the Graph he has to keep and what part it can dump.
                              NOTE: this variable is plural, because you can have multiple output nodes
    :param file_name:
    :return:
    """
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + file_name

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    print(input_checkpoint)
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def weighted_BCE_loss(predictions, labels, weight0=1, weight1=1):
    with tf.variable_scope("WCE_Loss"):
        predictions = tf.clip_by_value(t=predictions, clip_value_min=0, clip_value_max=1)
        labels = tf.clip_by_value(t=labels, clip_value_min=0, clip_value_max=1)

        inv_labels = 1 - labels
        coef0 = inv_labels * weight0
        coef1 = labels * weight1
        coefficient = coef0 + coef1
        label_shape = labels.get_shape().as_list()
        cross_entropy = - 1.0 * (tf.multiply(
                            labels * shortened_loss_function(predictions) +
                            inv_labels * log_loss_function(1 - predictions), coefficient))
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name="cross_entropy")
        ones_mean = tf.reduce_mean(cross_entropy * labels, name="ones_mean") * label_shape[
            0] / (tf.reduce_sum(labels) + 1e-10)
        zeros_mean = tf.reduce_mean(cross_entropy * inv_labels, name="zeros_mean") * label_shape[
            0] / (tf.reduce_sum(inv_labels) + 1e-10)

        return cross_entropy_mean, ones_mean, zeros_mean, cross_entropy


def log_loss_function(value):
    epsilon = tf.constant(math.e**(-5), dtype=tf.float32, name="epsilon")
    inner_log = tf.minimum(tf.pow(value, 1.5) + epsilon, (value + 1.0) / 2.0)
    return tf.log(inner_log)*focal_loss_modulation(value)


def shortened_loss_function(value):
    # loss(0) = -1
    # loss(1) = 0
    return tf.log(((1 + value * (math.e - 1)) / math.e))*focal_loss_modulation(value)


def focal_loss_modulation(value):
    return tf.pow(tf.clip_by_value(1-value, 0.0, 1.0), 0.5)


def weight_decay_loss():
    return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name="RegularizationLossSum")


def concat(values, axis, name="Concat"):
    global IS_OLD_TENSORFLOW

    if IS_OLD_TENSORFLOW:
        concatenated_values = tf.concat(values=values, concat_dim=axis, name=name)
    else:
        concatenated_values = tf.concat(values=values, axis=axis, name=name)
    return concatenated_values


def initialize_tf_variables():
    global IS_OLD_TENSORFLOW

    if IS_OLD_TENSORFLOW:
        return tf.initialize_all_variables()
    else:
        return tf.global_variables_initializer()


def create_summary_writer(logpath, sess):
    global IS_OLD_TENSORFLOW

    if IS_OLD_TENSORFLOW:
        return tf.train.SummaryWriter(logdir=logpath, graph=sess.graph)
    else:
        return tf.summary.FileWriter(logdir=logpath, graph=sess.graph)
