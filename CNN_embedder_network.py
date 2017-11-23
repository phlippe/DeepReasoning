import sys

import numpy as np

from ops import *


# TODO: Set instead over batch sizes over height. Might be faster...
class CNNEmbedder:
    def __init__(self, embedding_size, layer_number=3, channel_size=-1, kernel_size=5, batch_size=1, input_channels=-1,
                 char_number=50, name="CNNEmbedder", reuse_vocab=False, tensor_height=1):

        assert layer_number > 0, "Number of layers can not be negative nor 0"
        assert embedding_size > 0, "The embedding size can not be negative nor 0"
        assert kernel_size > 0, "The kernel size can not be negative nor 0"
        assert batch_size > 0, "The batch size can not be negative nor 0"
        assert tensor_height > 0, "The number of clauses, concatenated over height, must be greater one"

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
        self.char_number = char_number
        self.tensor_height = tensor_height

        self.vocab_table = None
        self.vocab_index_tensor = None
        self.vocab_offset = None
        self.input_clause = None
        self.input_length = None
        self.embedded_vector = None
        self.max_fun_code = None
        self.create_lookup_table(reuse_vocab=reuse_vocab)
        self.forward()

    def forward(self):
        with tf.variable_scope(self.name):
            self.input_clause = tf.placeholder(dtype="int32",
                                               shape=[self.batch_size * self.tensor_height, self.char_number],
                                               name="InputClause")
            self.input_length = tf.placeholder(dtype="int32",
                                               shape=[self.batch_size], name="InputClauseLength")
            all_vocabs = tf.reshape(tensor=self.input_clause, shape=[-1])
            vocab_indices = tf.gather(self.vocab_index_tensor, all_vocabs + self.vocab_offset)
            embedded_vocabs = tf.nn.embedding_lookup(params=self.vocab_table, ids=vocab_indices, name="Vocab_Lookup")
            if IS_OLD_TENSORFLOW:
                input_tensor = tf.pack(
                    values=tf.split(value=embedded_vocabs, num_split=self.batch_size * self.tensor_height, split_dim=0),
                    axis=0)
            else:
                input_tensor = tf.stack(
                    values=tf.split(value=embedded_vocabs, num_or_size_splits=self.batch_size * self.tensor_height,
                                    axis=0),
                    axis=0)
            # TODO: Check which input clause is where after reshaping
            input_tensor = tf.reshape(tensor=input_tensor,
                                      shape=[self.batch_size, self.tensor_height, -1, self.channel_size])

            first_layer = conv1d(input_=input_tensor, output_dim=self.channel_size, kernel_size=self.kernel_size,
                                 name=self.name + "_Conv1", relu=True)
            second_layer = conv1d(input_=first_layer, output_dim=self.channel_size, kernel_size=self.kernel_size,
                                  name=self.name + "_Conv2", relu=True)
            final_layer = conv1d(input_=second_layer, output_dim=self.embedding_size, kernel_size=self.kernel_size,
                                 name=self.name + "_Conv3", relu=True)

            self.channel_max_pool(final_layer)

    def channel_max_pool(self, final_layer):
        with tf.variable_scope("channel_max_pool"):
            if IS_OLD_TENSORFLOW:
                single_clauses = tf.unpack(value=final_layer, axis=0)
                self.embedded_vector = []
                for c in range(len(single_clauses)):
                    self.embedded_vector.append(
                        tf.reduce_max(input_tensor=single_clauses[c][:, :self.input_length[c], :],
                                      reduction_indices=1, keep_dims=True,
                                      name=self.name + "_MaxPool_" + str(c)))
                self.embedded_vector = tf.pack(values=self.embedded_vector, axis=0)
                # TF 0.11 does not support tensor indices with conformed shapes
                self.embedded_vector = tf.reshape(self.embedded_vector,
                                                  shape=[self.batch_size, self.tensor_height, 1, self.embedding_size],
                                                  name="EmbeddedVector")
            else:
                # self.embedded_vector = tf.reduce_max(input_tensor=final_layer, axis=2, keep_dims=True,
                #                                      name=self.name + "_MaxPool")
                single_clauses = tf.unstack(value=final_layer, axis=0)
                self.embedded_vector = []
                for c in range(len(single_clauses)):
                    self.embedded_vector.append(
                        tf.reduce_max(input_tensor=single_clauses[c][:, :self.input_length[c], :],
                                      axis=1, keep_dims=True,
                                      name=self.name + "_MaxPool_" + str(c)))
                self.embedded_vector = tf.stack(values=self.embedded_vector, axis=0, name="EmbeddedVector")

    def get_random_clause(self):
        random_clause = np.random.randint(0, len(list(self.get_vocabulary().values())),
                                          [self.batch_size * self.tensor_height, self.char_number], dtype=np.int32)
        random_clause = np.take(a=list(self.get_vocabulary().values()), indices=random_clause)
        return random_clause

    def get_zero_clause(self):
        zero_clause = np.zeros(shape=[self.batch_size * self.tensor_height, self.char_number], dtype=np.int32)
        return zero_clause

    def get_random_length(self):
        random_length = np.random.randint(5, self.char_number, [self.batch_size], dtype=np.int32)
        return random_length

    def create_lookup_table(self, reuse_vocab=False):
        with tf.variable_scope("Vocabulary", reuse=reuse_vocab):
            vocabulary = self.get_vocabulary()
            variable_keys = vocabulary.values()
            self.vocab_table = tf.get_variable("Vocabs", shape=[len(variable_keys), self.channel_size],
                                               dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer())

            self.vocab_index_tensor, self.vocab_offset = self.create_index_vector()

    def create_index_vector(self):
        vocabulary = self.get_vocabulary()
        fun_codes = vocabulary.values()
        if sys.version_info >= (3, 0):
            fun_codes = list(fun_codes)
        fun_codes_offset = - min(fun_codes)
        fun_codes = [fun_codes[i] + fun_codes_offset for i in range(len(fun_codes))]  # All greater than 0
        self.max_fun_code = max(fun_codes)

        index_values = np.zeros([max(fun_codes) + 1],
                                dtype=np.int32) - 1  # If unknown fun_code is given, -1 raises an error
        for i in range(len(fun_codes)):
            index_values[fun_codes[i]] = i

        return tf.constant(index_values, dtype=tf.int32, name="Index_Vector"), tf.constant(fun_codes_offset,
                                                                                           dtype=tf.int32,
                                                                                           name="Index_Offset")

    @staticmethod
    def get_vocabulary():
        with open('Vocabs.txt', 'r') as inf:
            dict_from_file = eval(inf.read())
        return dict_from_file
