from ops import *
from CNN_embedder_network import CNNEmbedder, NetType
from Comb_LSTM_network import CombLSTMNetwork

##########################
####      INPUTS      ####
##########################
INIT_CLAUSE_LENGTH = "InitClausesLength"
NEG_CONJ_FEATURE_INPUT = "NegConjFeaturesPlaceholder"
INIT_STATE_FEATURE_INPUT = "InitStateFeaturesPlaceholder"

##########################
####      OUTPUTS     ####
##########################
INIT_STATE_OUTPUT = "InitStateFeatures"
NEG_CONJ_OUTPUT = "NegConjFeatures"
CLAUSE_OUTPUT = "ClauseFeatures"
WEIGHT_OUTPUT = "FinalWeights"

##########################
####   INNER LAYERS   ####
##########################
FC_LAYER_1 = "Comb_ClauseNegConj"
FC_LAYER_2 = "Comb_InitMemory"
FC_LAYER_3 = "Comb_final"


class CombLSTMInterference:
    def __init__(self, embedding_size=1024, batch_size=32, name="CombLSTMNet",
                 comb_features=1024, embedding_net_type=NetType.DILATED_DENSE_BLOCK,
                 use_conversion=False):

        assert embedding_size > 0, "Number of channels for first layer has to be greater than 0!"

        self.num_init_clauses = batch_size
        self.num_train_clauses = batch_size
        self.batch_size = batch_size
        self.comb_features = comb_features
        self.name = name
        self.tensor_height = 1
        self.embedding_size = embedding_size
        self.embedding_net_type = embedding_net_type
        self.use_conversion = use_conversion
        self.wavenet_blocks = 1
        self.wavenet_layers = 2

        self.labels = None
        self.weight = None
        self.loss = None
        self.loss_ones = None
        self.loss_zeros = None
        self.loss_regularization = None
        self.all_losses = None
        self.init_clauses = None
        self.train_clauses = None
        self.clause_embedder = None
        self.neg_conjecture_embedder = None
        self.neg_conj_embedded = None
        self.init_clauses_length = None
        self.lstm_one = None
        self.lstm_initial = None
        self.state_lstm_one = None
        self.state_lstm_initial = None
        self.forward()

    def forward(self):
        with tf.variable_scope(self.name):
            self.labels = tf.placeholder(dtype="float32", shape=[self.batch_size], name="Labels")

            self.embed_clauses()
            self.embed_neg_conjecture()

            self.run_init_clauses()
            self.run_comb_layers()

    def run_comb_layers(self):
        global FC_LAYER_2, FC_LAYER_3
        with tf.name_scope("CombNetwork"):

            neg_conj_placeholder = tf.placeholder(dtype=tf.float32, shape=self.neg_conj_embedded.shape,
                                                  name=NEG_CONJ_FEATURE_INPUT)
            shaped_state = tf.placeholder(dtype=tf.float32, shape=self.state_lstm_initial.shape,
                                          name=INIT_STATE_FEATURE_INPUT)

            neg_conj_vector = tf.tile(tf.expand_dims(neg_conj_placeholder, axis=0), multiples=[self.num_train_clauses, 1])
            layer_comb = self.first_combination_layer(self.train_clauses, neg_conj_vector, reuse=True)

            shaped_state = tf.nn.tanh(shaped_state, name="StateActivationFct")
            shaped_state = tf.tile(tf.expand_dims(shaped_state, axis=0), multiples=[self.num_train_clauses, 1])
            tensor_with_state = tf.concat(values=[layer_comb, shaped_state], axis=1)
            tensor_with_state = tf.reshape(tensor_with_state, shape=[self.num_train_clauses, 1, 1, self.comb_features*2])

            layer_initial = fully_connected(input_=tensor_with_state, outputs=self.comb_features,
                                            activation_fn=tf.nn.relu, reuse=False, name=FC_LAYER_2,
                                            use_batch_norm=False)
            self.weight = fully_connected(input_=layer_initial, outputs=1, activation_fn=tf.nn.sigmoid,
                                          reuse=False, name=FC_LAYER_3, use_batch_norm=False)
            self.weight = tf.squeeze(self.weight, name=WEIGHT_OUTPUT)

    def embed_clauses(self):
        self.clause_embedder = CNNEmbedder(embedding_size=self.embedding_size, name="ClauseEmbedder",
                                           batch_size=self.batch_size,
                                           char_number=None, net_type=self.embedding_net_type, tensor_height=1,
                                           use_batch_norm=False, wavenet_blocks=self.wavenet_blocks,
                                           wavenet_layers=self.wavenet_layers, dropout_rate=0.0,
                                           is_training=False, use_conversion=self.use_conversion)
        # Squeeze so that LSTMs can run with fully connected layers
        all_clauses = tf.squeeze(self.clause_embedder.embedded_vector, name=CLAUSE_OUTPUT)
        self.train_clauses = all_clauses
        self.init_clauses = all_clauses

    def run_init_clauses(self):
        # Number of real init clauses that are given. Maximum is "self.num_init_clauses".
        # Other places must be filled up with default/random clause
        with tf.name_scope("InitialClauses"):
            self.init_clauses_length = tf.placeholder(dtype="int32", shape=[1], name=INIT_CLAUSE_LENGTH)

            with tf.name_scope("FirstCombLayer"):
                # Use output of first LSTM to determine the states of the second one
                # Run through first fully connected layer
                neg_conj_vector = tf.tile(input=tf.expand_dims(self.neg_conj_embedded, axis=0),
                                          multiples=[self.num_init_clauses, 1])
                comb_layer = self.first_combination_layer(self.init_clauses, neg_conj_vector, reuse=False)

            with tf.variable_scope("LSTM_INITIAL"):
                with tf.name_scope("Preparation"):
                    # Prepare batch for LSTM. New shape: [Time, Clause-Features]
                    # Split over time
                    time_batches = tf.split(value=comb_layer, num_or_size_splits=comb_layer.shape[0], axis=0)

                    self.lstm_initial = tf.contrib.rnn.BasicLSTMCell(self.comb_features)
                    hidden_state_initial = tf.zeros(shape=[1, self.comb_features])
                    self.state_lstm_initial = hidden_state_initial, hidden_state_initial

                with tf.name_scope("LSTM"):
                    # Run second LSTM over all time steps. Output will not be used anymore
                    all_states = []
                    for batch in time_batches:
                        _, self.state_lstm_initial = self.lstm_initial(batch,
                                                                       self.tuple_to_lstm_state(self.state_lstm_initial))
                        all_states.append(self.state_lstm_initial)

                with tf.name_scope("FinalState"):
                    self.state_lstm_initial = self.extract_state(all_states)
                    self.state_lstm_initial = tf.squeeze(self.state_lstm_initial, name=INIT_STATE_OUTPUT)

    def extract_state(self, all_states):
        with tf.name_scope("ExtractStates"):
            # Extract these states that have only seen real initial clauses and no default ones
            chosen_states_h = self.short_state_extraction(all_states, state_index=0)
            return chosen_states_h

    def tuple_to_lstm_state(self, state_tuple):
        with tf.name_scope("StateConversion"):
            return state_tuple[0], tf.nn.tanh(state_tuple[0])   # tf.stop_gradient(tf.nn.tanh(state_tuple[0]))

    def short_state_extraction(self, all_states, state_index):
        state_tensor = tf.stack(values=[a[state_index] for a in all_states], axis=0)
        return state_tensor[self.init_clauses_length[0] - 1, :, :]

    def first_combination_layer(self, clause_vector, neg_conj_vector, reuse=False):
        global FC_LAYER_1
        concatenated_features = concat(
            values=[clause_vector, neg_conj_vector], axis=1,
            name="ConcatClauseNegConj")
        concatenated_features = tf.reshape(tensor=concatenated_features, shape=[self.batch_size, 1, 1, self.comb_features*2])
        print("Concat feature shape: "+str(concatenated_features.get_shape().as_list()))

        comb_layer = fully_connected(concatenated_features, self.comb_features, activation_fn=tf.nn.relu, reuse=reuse,
                                     name=FC_LAYER_1, use_batch_norm=False)
        return tf.squeeze(comb_layer)

    def embed_neg_conjecture(self):
        self.neg_conjecture_embedder = CNNEmbedder(embedding_size=self.embedding_size, name="NegConjectureEmbedder",
                                                   reuse_vocab=True, batch_size=1, char_number=None,
                                                   net_type=self.embedding_net_type, tensor_height=1,
                                                   use_batch_norm=False, wavenet_blocks=self.wavenet_blocks,
                                                   wavenet_layers=self.wavenet_layers, dropout_rate=0.0,
                                                   is_training=False, use_conversion=self.use_conversion)
        # Squeeze so that LSTMs can run with fully connected layers
        self.neg_conj_embedded = tf.squeeze(self.neg_conjecture_embedder.embedded_vector, name=NEG_CONJ_OUTPUT)


