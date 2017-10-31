import numpy as np

from ops import *


class CNNEmbedder:
    def __init__(self, embedding_size, layer_number=3, channel_size=-1, kernel_size=5, batch_size=1, input_channels=-1,
                 name="CNNEmbedder"):

        assert layer_number > 0, "Number of layers can not be negative nor 0"
        assert embedding_size > 0, "The embedding size can not be negative nor 0"
        assert kernel_size > 0, "The kernel size can not be negative nor 0"
        assert batch_size > 0, "The batch size can not be negative nor 0"

        self.layer_number = layer_number
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.name = name
        if channel_size <= -1:
            self.channel_size = embedding_size
        else:
            self.channel_size = channel_size
        if input_channels <= -1:
            self.input_channels = embedding_size
        else:
            self.input_channels = input_channels

        self.input_clause = None
        self.embedded_vector = None
        self.forward()

    def forward(self):
        with tf.variable_scope(self.name):
            self.input_clause = tf.placeholder(dtype="float32", shape=[self.batch_size, 1, None, self.input_channels],
                                               name="InputClause")
            first_layer = conv1d(input_=self.input_clause, output_dim=self.channel_size, kernel_size=self.kernel_size,
                                 name=self.name + "_Conv1", relu=True)
            second_layer = conv1d(input_=first_layer, output_dim=self.channel_size, kernel_size=self.kernel_size,
                                  name=self.name + "_Conv2", relu=True)
            final_layer = conv1d(input_=second_layer, output_dim=self.embedding_size, kernel_size=self.kernel_size,
                                 name=self.name + "_Conv3", relu=True)
            self.embedded_vector = tf.reduce_max(input_tensor=final_layer, axis=2, keep_dims=True,
                                                 name=self.name + "_MaxPool")

    def get_random_clause(self):
        return np.random.rand(self.batch_size, 1, np.random.randint(5, 10), self.channel_size)
