import sys

import numpy as np
from create_conversion_vocab import generate_vocab_variables

from ops import *
from enum import Enum


class NetType(Enum):
    STANDARD = 0,
    SMALL_WAVENET = 1,
    WAVENET_BLOCKS = 2,
    DILATED_DENSE_BLOCK = 3

GLOBAL_VOCAB = None
GLOBAL_ARITIES = None

class CNNEmbedder:
    def __init__(self, embedding_size, layer_number=3, channel_size=-1, kernel_size=5, batch_size=1, input_channels=-1,
                 char_number=50, name="CNNEmbedder", reuse_vocab=False, tensor_height=1, net_type=NetType.STANDARD,
                 reuse_weights=False, use_batch_norm=False,  wavenet_blocks=1, wavenet_layers=2, dropout_rate=0.5,
                 is_training=False, max_pool_prop=0.9, use_conversion=False):

        assert layer_number > 0, "Number of layers can not be negative nor 0"
        assert embedding_size > 0, "The embedding size can not be negative nor 0"
        assert kernel_size > 0, "The kernel size can not be negative nor 0"
        assert batch_size > 0, "The batch size can not be negative nor 0"
        assert tensor_height > 0, "The number of clauses, concatenated over height, must be greater one"
        assert type(net_type) is NetType, "Network type must be a enum instance"

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
        self.net_type = net_type
        self.wavenet_blocks = wavenet_blocks
        self.wavenet_layers = wavenet_layers
        self.reuse_weights = reuse_weights
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.max_pool_prop = max_pool_prop
        self.use_conversion = use_conversion

        self.vocab_table = None
        self.arity_table = None
        self.vocab_index_tensor = None
        self.arity_index_tensor = None
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
            # self.input_clause = tf.Print(self.input_clause, [self.input_clause, self.input_length], message="Input: ", summarize=8)
            embedded_vocabs = self.embed_input_clause()
            # embedded_vocabs = tf.Print(embedded_vocabs, [embedded_vocabs], message="Vocabs: ", summarize=8)
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

            if self.net_type == NetType.STANDARD:
                print("Build up standard network...")
                first_layer = conv1d(input_=input_tensor, output_dim=self.channel_size, kernel_size=self.kernel_size,
                                     name=self.name + "_Conv1", relu=True, use_batch_norm=self.use_batch_norm,
                                     reuse=self.reuse_weights)
                second_layer = conv1d(input_=first_layer, output_dim=self.channel_size, kernel_size=self.kernel_size,
                                      name=self.name + "_Conv2", relu=True, use_batch_norm=self.use_batch_norm,
                                      reuse=self.reuse_weights)
                final_layer = conv1d(input_=second_layer, output_dim=self.embedding_size, kernel_size=self.kernel_size,
                                     name=self.name + "_Conv3", relu=True, use_batch_norm=self.use_batch_norm,
                                     reuse=self.reuse_weights)
            elif self.net_type == NetType.SMALL_WAVENET:
                print("Build up small wavenet architecture...")
                first_layer = wavenet_layer(input_=input_tensor, kernel_size=self.kernel_size, dilation_rate=1,
                                            name=self.name + "_Conv1", reuse=self.reuse_weights)
                second_layer = wavenet_layer(input_=first_layer, kernel_size=self.kernel_size, dilation_rate=2,
                                             name=self.name + "_Conv2", reuse=self.reuse_weights)
                final_layer = wavenet_layer(input_=second_layer, kernel_size=self.kernel_size, dilation_rate=1,
                                            name=self.name + "_Conv3", reuse=self.reuse_weights)
            elif self.net_type == NetType.WAVENET_BLOCKS:
                print("Build up wavenet blocks...")
                final_layer = hierarchical_wavenet_block(input_tensor=input_tensor, block_number=self.wavenet_blocks,
                                                         layer_number=self.wavenet_layers,
                                                         kernel_size=3, dropout_rate=0.2, reuse=self.reuse_weights,
                                                         name=self.name + "_Wavenet_Block")
            elif self.net_type == NetType.DILATED_DENSE_BLOCK:
                print("Build up dilated dense block...")
                dense_layer = dilated_dense_block(input_tensor=input_tensor, layer_number=5,
                                                  channel_size=self.embedding_size, end_channels=2*self.embedding_size,
                                                  kernel_size=3, training=self.is_training,
                                                  dropout_rate=self.dropout_rate)

                # final 3x1 convolution without stride to support higher complexity for small clauses
                final_layer = conv1d(input_=dense_layer, output_dim=2*self.embedding_size, kernel_size=3,
                                     name="FinalLocalConv", relu=False, use_batch_norm=self.use_batch_norm,
                                     reuse=self.reuse_weights)
                tf.summary.scalar(name="FinalLayer_Mean", tensor=tf.reduce_mean(final_layer))
                tf.summary.scalar(name="FinalLayer_Max", tensor=tf.reduce_max(final_layer))
                tf.summary.scalar(name="FinalLayer_Min", tensor=tf.reduce_min(final_layer))
                final_layer = tf.nn.tanh(final_layer)   # Normalizing input between -1 and 1

            else:
                print(" [!] ERROR: Unknown network type")
                sys.exit(1)
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
                    current_clause = single_clauses[c][:, :self.input_length[c], :]
                    max_feature = tf.reduce_max(input_tensor=current_clause,
                                                axis=1, keep_dims=True,
                                                name=self.name + "_MaxPool_" + str(c))
                    mean_feature = tf.reduce_mean(input_tensor=current_clause,
                                                  axis=1, keep_dims=True,
                                                  name=self.name + "_MeanPool_" + str(c))
                    mean_feature = tf.where(tf.is_nan(mean_feature), tf.zeros_like(mean_feature), mean_feature)
                    # combined_feature = self.max_pool_prop * max_feature + (1 - self.max_pool_prop) * mean_feature
                    combined_feature = max_feature
                    self.embedded_vector.append(combined_feature)
                self.embedded_vector = tf.stack(values=self.embedded_vector, axis=0, name="EmbeddedVector")

    def embed_input_clause(self):
        all_vocabs = tf.reshape(tensor=self.input_clause, shape=[-1])
        # self.vocab_index_tensor = tf.Print(self.vocab_index_tensor, [self.vocab_index_tensor], message="Index tensor: ", summarize=8)
        vocab_indices = tf.gather(self.vocab_index_tensor, all_vocabs + self.vocab_offset)
        arity_indices = tf.gather(self.arity_index_tensor, all_vocabs)
        # vocab_indices = tf.Print(vocab_indices, [vocab_indices, arity_indices], message="Indices: ", summarize=8)
        embedded_vocabs = tf.nn.embedding_lookup(params=self.vocab_table, ids=vocab_indices, name="Vocab_Lookup")
        embedded_arities = tf.nn.embedding_lookup(params=self.arity_table, ids=arity_indices, name="Arity_Lookup")
        return tf.concat([embedded_vocabs, embedded_arities], axis=1)

    def get_random_clause(self):
        random_clause = np.random.randint(0, len(list(self.get_vocabulary(use_conversion=self.use_conversion).values())),
                                          [self.batch_size * self.tensor_height, self.char_number], dtype=np.int32)
        random_clause = np.take(a=list(self.get_vocabulary(use_conversion=self.use_conversion).values()), indices=random_clause)
        return random_clause

    def get_zero_clause(self):
        zero_clause = np.zeros(shape=[self.batch_size * self.tensor_height, self.char_number], dtype=np.int32)
        return zero_clause

    def get_random_length(self):
        random_length = np.random.randint(5, self.char_number, [self.batch_size], dtype=np.int32)
        return random_length

    def create_lookup_table(self, reuse_vocab=False):
        global GLOBAL_ARITIES, GLOBAL_VOCAB
        with tf.variable_scope("Vocabulary", reuse=reuse_vocab):
            vocab_values, keys, vocab_offset = self.get_vocab_values_keys()
            self.vocab_index_tensor, self.vocab_offset = self.create_index_vector()
            self.arity_index_tensor, max_arity = self.create_arity_vector()

            # self.vocab_table = tf.get_variable("Vocabs",
            #                                    shape=[max(vocab_values) + vocab_offset, self.channel_size / 2],
            #                                    dtype=tf.float32,
            #                                    initializer=tf.contrib.layers.xavier_initializer())
            # self.arity_table = tf.get_variable("Arities", shape=[max_arity, self.channel_size / 2],
            #                                    dtype=tf.float32,
            #                                    initializer=tf.contrib.layers.xavier_initializer())
            # print(vocab_shape)
            if reuse_vocab:
                self.vocab_table = tf.stop_gradient(GLOBAL_VOCAB, name="StopVocabGradients")
                self.arity_table = tf.stop_gradient(GLOBAL_ARITIES, name="StopAritiesGradients")
            else:
                vocab_shape = [max(vocab_values) + vocab_offset, int(self.channel_size / 2)]
                arity_shape = [max_arity, int(self.channel_size / 2)]

                if not self.use_conversion:
                    print("No conversion used")
                    self.vocab_table = get_vocab_variable(name="Vocabs", shape=vocab_shape)
                    self.arity_table = get_vocab_variable(name="Arities", shape=arity_shape)
                else:
                    print("Use conversion")
                    voc_vars = np.array(generate_vocab_variables(vocab_shape[0], vocab_shape[1],
                                                                 min_diffs=int(vocab_shape[1] / 2),
                                                                 min_commons=int(vocab_shape[1] / 4)),
                                        dtype=np.float32)
                    self.vocab_table = tf.get_variable(name="Vocabs",
                                                       initializer=tf.constant(voc_vars),
                                                       trainable=False,
                                                       dtype=tf.float32)

                    self.arity_table = get_vocab_variable(name="Arities", shape=arity_shape)

                    # arity_vars = np.array(generate_vocab_variables(arity_shape[0], arity_shape[1],
                    #                                                min_diffs=int(arity_shape[1] / 2),
                    #                                                min_commons=int(arity_shape[1] / 4)),
                    #                       dtype=np.float32)
                    # self.arity_table = tf.get_variable(name="Arities",
                    #                                    initializer=tf.constant(arity_vars),
                    #                                    trainable=False,
                    #                                    dtype=tf.float32)
                self.vocab_table = tf.add(self.vocab_table, 0.0, name="PreMultVocab")
                self.arity_table = tf.add(self.arity_table, 0.0, name="PreMultArity")
                GLOBAL_VOCAB = self.vocab_table
                GLOBAL_ARITIES = self.arity_table

    def create_index_vector(self):
        fun_codes, _, fun_codes_offset = self.get_vocab_values_keys()
        self.max_fun_code = max(fun_codes)

        index_values = np.zeros([max(fun_codes) + 1],
                                dtype=np.int32)  # -1 If unknown fun_code is given, -1 raises an error
        for i in range(len(fun_codes)):
            index_values[fun_codes[i]] = i
        print("Indices: "+str(np.where(index_values == 0)))

        return tf.constant(index_values, dtype=tf.int32, name="Index_Vector"), tf.constant(fun_codes_offset,
                                                                                           dtype=tf.int32,
                                                                                           name="Index_Offset")

    def create_arity_vector(self):
        values, keys, _ = self.get_vocab_values_keys()
        arities = [CNNEmbedder.get_arity_from_vocab(x) for x in keys]
        arities_offset = min(arities)
        arities = [arities[i] - arities_offset for i in range(len(arities))]
        vector = np.zeros([max(values) + 1], dtype=np.int32)

        for i in range(len(values)):
            vector[values[i]] = arities[i]

        return tf.constant(vector, dtype=tf.int32, name="Arity_Index_Vector"), max(arities)

    def get_vocab_values_keys(self):
        vocabulary = CNNEmbedder.get_vocabulary(use_conversion=self.use_conversion)
        values = vocabulary.values()
        keys = vocabulary.keys()
        if sys.version_info >= (3, 0):
            values = list(values)
            keys = list(keys)
        values_offset = - min(values)
        values = [values[i] + values_offset for i in range(len(values))]  # All greater than 0
        values, keys = (list(t) for t in zip(*sorted(zip(values, keys))))
        print("Vocab values: "+str(values)+", Keys: "+str(keys[:10]))
        return values, keys, values_offset

    @staticmethod
    def get_arity_from_vocab(vocab):
        splitted_vocab = vocab.rsplit('#', 2)
        if len(splitted_vocab) > 1:
            return int(splitted_vocab[-1])
        else:
            if vocab[0] == 'X':
                return -1
            else:
                if vocab == " ":
                    return -3
                else:
                    return -2

    @staticmethod
    def get_vocabulary(use_conversion):
        if use_conversion:
            VOCAB_FILE_NAME = "Conversion_vocab.txt"
        else:
            VOCAB_FILE_NAME = "Vocabs.txt"
        with open(VOCAB_FILE_NAME, 'r') as inf:
            dict_from_file = eval(inf.read())
        return dict_from_file
