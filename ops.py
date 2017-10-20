import os

import tensorflow as tf


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=1, d_w=1,
           name="conv2d", reuse=False, padding='SAME'):
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
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

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
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.atrous_conv2d(
            input_, w, rate=dilation_rate, padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def conv1d(input_, output_dim, kernel_size=3, d_h=1, d_w=1,
           name="conv1d", reuse=False, padding='SAME'):
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
    :return: Output of convolution with shape NHWC (channel size = output_dim)
    """
    input_shape = input_.get_shape()
    if len(input_shape) == 3:
        input_ = tf.reshape(input_, shape=(input_shape[0], 1, input_shape[1], input_shape[2]))
    return conv2d(input_=input_, output_dim=output_dim, k_h=1, k_w=kernel_size, d_h=d_h, d_w=d_w,
                  name=name, reuse=reuse, padding=padding)


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
    input_shape = input_.get_shape()
    if len(input_shape) == 3:
        input_ = tf.reshape(input_, shape=(input_shape[0], 1, input_shape[1], input_shape[2]))
    return dilated_conv2d(input_=input_, output_dim=output_dim, k_h=1, k_w=k_w, dilation_rate=dilation_rate,
                          name=name, reuse=reuse, padding=padding)


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
        input_shape = input_.get_shape()
        channel_size = input_shape[-1]
        filter_layer = tf.nn.tanh(
            dilated_conv1d(input_=input_, output_dim=channel_size, k_w=kernel_size, dilation_rate=dilation_rate,
                           name=name + "_filter", reuse=reuse))
        gate_layer = tf.nn.sigmoid(
            dilated_conv1d(input_=input_, output_dim=channel_size, k_w=kernel_size, dilation_rate=dilation_rate,
                           name=name + "_gate", reuse=reuse))
        layer_d = filter_layer * gate_layer + input_
        return layer_d


def dropout(input_, p):
    """
    Dropout layer.
    :param input_: Input to dropout
    :param p: Probability with which a node is set to zero
    :return: Input with applied dropout
    """
    return tf.nn.dropout(input_, p)


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
