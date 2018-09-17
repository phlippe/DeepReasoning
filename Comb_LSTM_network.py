from ops import *
from CNN_embedder_network import CNNEmbedder, NetType
from random import shuffle
import math
import itertools

FC_LAYER_1 = "Comb_ClauseNegConj"
FC_LAYER_2 = "Comb_InitMemory"
FC_LAYER_3 = "Comb_final"


class CombLSTMNetwork:
    def __init__(self, embedding_size=1024, num_init_clauses=32, num_proof=4, num_train_clauses=32, num_shuffles=4,
                 weight0=1, weight1=1, name="CombLSTMNet", wavenet_blocks=1, wavenet_layers=2, comb_features=1024,
                 embedding_net_type=NetType.STANDARD, dropout_rate_embedder=0.2, dropout_rate_fc=0.0,
                 use_conversion=False, dropout_rate_neg_conj=0.125):

        assert embedding_size > 0, "Number of channels for first layer has to be greater than 0!"

        self.num_init_clauses = num_init_clauses
        self.num_proof = num_proof
        self.num_train_clauses = num_train_clauses
        self.batch_size = num_train_clauses * num_proof
        self.num_shuffles = num_shuffles
        self.comb_features = comb_features
        self.name = name
        self.tensor_height = 1
        self.embedding_size = embedding_size
        self.weight0 = weight0
        self.weight1 = weight1
        self.wavenet_blocks = wavenet_blocks
        self.wavenet_layers = wavenet_layers
        self.embedding_net_type = embedding_net_type
        self.dropout_rate_embedder = dropout_rate_embedder
        self.dropout_rate_fc = dropout_rate_fc
        self.dropout_rate_neg_conj = dropout_rate_neg_conj
        self.use_conversion = use_conversion

        self.labels = None
        self.is_training = None
        self.weight = None
        self.loss = None
        self.loss_ones = None
        self.loss_zeros = None
        self.loss_regularization = None
        self.loss_euclidian = None
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
            self.is_training = tf.placeholder_with_default(True, shape=())

            self.embed_clauses()
            self.embed_neg_conjecture()

            self.run_init_clauses()
            self.run_comb_layers()

    def run_comb_layers(self):
        global FC_LAYER_2, FC_LAYER_3
        with tf.name_scope("CombNetwork"):

            neg_conj_vector = self.repeat_tensor(tensor_to_repeat=self.neg_conj_embedded, axis=0,
                                                 times=self.num_train_clauses)
            # =======================
            # === NEGCONJ DROPOUT ===
            # =======================
            # neg_conj_dropout_probs = tf.random_uniform(shape=[tf.shape(neg_conj_vector)[0], 1], minval=0.0, maxval=1.0, name="NegConjDropoutProbs")
            # neg_conj_dropout_mask = tf.cast(tf.less_equal(neg_conj_dropout_probs, self.dropout_rate_neg_conj), dtype=tf.float32) * tf.cast(self.is_training, dtype=tf.float32)
            # neg_conj_dropout_mask = tf.Print(neg_conj_dropout_mask, [neg_conj_dropout_mask], message="Dropout mask: ", summarize=16)
            # neg_conj_vector = neg_conj_vector * (1 - neg_conj_dropout_mask)
            # neg_conj_vector = tf.Print(neg_conj_vector, [neg_conj_vector[:,0], self.train_clauses[:,0]], summarize=16, message="Applying dropout: ")
            # self.train_clauses = self.train_clauses * (1 + neg_conj_dropout_mask) # Scale up by 2 if neg conj is dropped
            # neg_conj_vector = tf.Print(neg_conj_vector, [self.train_clauses[:,0]], summarize=16, message="Scaling: ")

            layer_comb = self.first_combination_layer(self.train_clauses, neg_conj_vector, reuse=True)
            layer_comb_dropout = dropout(layer_comb, self.dropout_rate_fc, training=self.is_training)
            # self.state_lstm_initial[0] = tf.Print(self.state_lstm_initial[0], [self.state_lstm_initial[0][:, 0]], message="State before: ", summarize=64)
            shaped_state = self.repeat_lstm_states(self.state_lstm_initial)[0]
            # shaped_state = tf.Print(shaped_state, [shaped_state[:, 0]], message="State after: ", summarize=64)
            shaped_state = tf.nn.tanh(shaped_state, name="StateActivationFct")
            tensor_with_state = tf.concat(values=[layer_comb_dropout, shaped_state], axis=1)
            layer_initial = fully_connected(input_=tensor_with_state, outputs=self.comb_features,
                                            activation_fn=tf.nn.relu, reuse=False, name=FC_LAYER_2,
                                            use_batch_norm=False)
            layer_initial_dropout = dropout(layer_initial, self.dropout_rate_fc, training=self.is_training)

            self.weight = fully_connected(input_=layer_initial_dropout, outputs=1, activation_fn=tf.nn.sigmoid,
                                          reuse=False, name=FC_LAYER_3, use_batch_norm=False)
            self.weight = tf.squeeze(self.weight, name="CalcWeights")
            if self.tensor_height != 1:
                self.weight = tf.reshape(tensor=self.weight, shape=[-1], name="ReshapeTo1D")
            self.loss, self.loss_ones, self.loss_zeros, self.all_losses = weighted_BCE_loss(self.weight, self.labels,
                                                                                            self.weight0, self.weight1)
            self.loss_regularization = weight_decay_loss()
            self.loss_euclidian = self.state_euclidian_loss(self.state_lstm_initial[0]) * 0.01
            self.loss += self.loss_regularization
            self.loss += self.loss_euclidian

            with tf.name_scope("SummaryVisu"):
                self.add_feature_visualizations([(self.neg_conj_embedded[0, :], "FeatureNegConj")])
                self.add_feature_visualizations([(self.train_clauses, "FeaturesClause"),
                                                 (layer_comb, "FeaturesCombClauseNegConj"),
                                                 (layer_initial, "FeaturesCombInitialMemory"),
                                                 (shaped_state, "FeaturesState")],
                                                indices=[(self.num_init_clauses, "Positive"),
                                                         (self.num_init_clauses+self.num_train_clauses/2, "Negative")])

    def state_euclidian_loss(self, states):
        states_per_proof = tf.split(states, num_or_size_splits=self.num_proof)
        rolled_states = list()
        for proof_index in range(self.num_proof):
            rolled_states.append(tf.concat([states_per_proof[proof_index][1:], states_per_proof[proof_index][0:1]], axis=0))
        rolled_states_tensor = tf.concat(rolled_states, axis=0)
        euclidian_distance = self.euclidian_distance(rolled_states_tensor, states)
        # euclidian_distance = tf.Print(euclidian_distance, [states[:, 0], rolled_states_tensor[:, 0]], message="Rolled tensor: ", summarize=16)
        euclidian_distance = euclidian_distance / (self.num_proof + self.num_shuffles)
        return euclidian_distance

    def euclidian_distance(self, feature_vector_0, feature_vector_1):
        return tf.sqrt(tf.reduce_sum(tf.square(feature_vector_0 - feature_vector_1)))

    def add_feature_visualizations(self, feature_tensor_tuples, scale_size=32, indices=None):
        if indices is None:
            for feature_tensor, name in feature_tensor_tuples:
                self.visualize_feature_tensor(feature_tensor, name, scale_size)
        else:
            for tensor_index, name_prefix in indices:
                tensor_index = int(tensor_index)
                with tf.name_scope(name_prefix):
                    for feature_tensor, name in feature_tensor_tuples:
                        self.visualize_feature_tensor(feature_tensor[tensor_index], name, scale_size)

    def visualize_feature_tensor(self, feature_tensor, name, scale_size):
        image = tf.reshape(feature_tensor, shape=[1, int(scale_size), int(self.comb_features / scale_size), 1])
        image = tf.tile(image, multiples=[1, 1, 1, 3])
        tf.summary.image(name=name,
                         tensor=image,
                         max_outputs=1)

    def repeat_lstm_states(self, state):
        rep_factor = int(self.num_train_clauses / self.num_shuffles)
        print("Repeat factor: " + str(rep_factor))
        print("Hidden states: " + str(state[0].get_shape().as_list()))
        print("Current states: " + str(state[1].get_shape().as_list()))
        repeated_hidden_states = self.repeat_initial_state(state[0]) # CombLSTMNetwork.repeat_tensor(tensor_to_repeat=state[0], axis=0, times=rep_factor)
        repeated_current_states = self.repeat_initial_state(state[1]) # CombLSTMNetwork.repeat_tensor(tensor_to_repeat=state[1], axis=0, times=rep_factor)
        print("Repeated hidden states: " + str(repeated_hidden_states.get_shape().as_list()))
        print("Repeated current states: " + str(repeated_current_states.get_shape().as_list()))
        return [repeated_hidden_states, repeated_current_states]

    def embed_clauses(self):
        self.clause_embedder = CNNEmbedder(embedding_size=self.embedding_size, name="ClauseEmbedder",
                                           batch_size=self.batch_size + self.num_init_clauses * self.num_proof,
                                           char_number=None, net_type=self.embedding_net_type, tensor_height=1,
                                           use_batch_norm=False, wavenet_blocks=self.wavenet_blocks,
                                           wavenet_layers=self.wavenet_layers, dropout_rate=self.dropout_rate_embedder,
                                           is_training=self.is_training, use_conversion=self.use_conversion)
        # Squeeze so that LSTMs can run with fully connected layers
        all_clauses = tf.split(tf.squeeze(self.clause_embedder.embedded_vector), num_or_size_splits=2, axis=0)
        self.train_clauses = all_clauses[0]
        self.init_clauses = all_clauses[1]

    def run_init_clauses(self):
        # Number of real init clauses that are given. Maximum is "self.num_init_clauses".
        # Other places must be filled up with default/random clause
        with tf.name_scope("InitialClauses"):
            self.init_clauses_length = tf.placeholder(dtype="int32", shape=[self.num_proof], name="InitClausesLength")

            with tf.name_scope("FirstCombLayer"):
                # Use output of first LSTM to determine the states of the second one
                # Run through first fully connected layer
                neg_conj_vector = self.repeat_tensor(tensor_to_repeat=self.neg_conj_embedded,
                                                     times=self.num_init_clauses,
                                                     axis=0)
                # neg_conj_vector = tf.Print(neg_conj_vector, [self.init_clauses_length], message="Init Length: ", summarize=6)
                comb_layer = self.first_combination_layer(self.init_clauses, neg_conj_vector, reuse=False)

            with tf.name_scope("ShuffleInitClauses"):
                # Split clauses belonging to different proofs/negated conjectures
                splitted_init_clauses = tf.split(value=comb_layer, num_or_size_splits=self.num_proof, axis=0)
                # Create randomly shuffled index matrices for all init clause lengths
                shuffle_list = CombLSTMNetwork.create_shuffle_tensor(self.num_init_clauses, self.num_shuffles)
                shuffle_tensor = tf.constant(value=shuffle_list, dtype="int32")
                # Randomly shuffle init clauses. For better generalization do it "self.num_shuffles" times differently.
                """
                Options for shuffling:
                1) tf.random_shuffle -> Problem: no gradients defined
                splitted_init_clauses = [
                    [tf.concat(values=[tf.random_shuffle(value=splitted_init_clauses[i][:self.init_clauses_length[i]]),
                                       splitted_init_clauses[i][self.init_clauses_length[i]:]],
                               axis=0)
                     for _ in range(self.num_shuffles)]
                    for i in range(self.num_proof)]
                    
                2) No shuffle at all -> Problem: bad generalization
                splitted_init_clauses = [
                     [splitted_init_clauses[i]
                     for _ in range(self.num_shuffles)]
                    for i in range(self.num_proof)]
                    
                3) Hard coded shuffle -> Best alternative
                splitted_init_clauses = [
                    [tf.gather(params=splitted_init_clauses[proof_index],
                               indices=shuffle_tensor[shuffle_index, self.init_clauses_length[proof_index]-1])
                     for shuffle_index in range(self.num_shuffles)]
                    for proof_index in range(self.num_proof)]
                """
                splitted_init_clauses = [
                    [tf.gather(params=splitted_init_clauses[proof_index],
                               indices=shuffle_tensor[shuffle_index, self.init_clauses_length[proof_index]-1])
                     for shuffle_index in range(self.num_shuffles)]
                    for proof_index in range(self.num_proof)]  # Shuffle init clauses for better generalization
                splitted_init_clauses = [tensor for sublist in splitted_init_clauses for tensor in
                                         sublist]  # Flatten list

            # with tf.variable_scope("LSTM_ONE"):
            #     # Prepare batch for LSTM. New shape: [Time, Proofs/Clauses, Clause-Features]
            #     init_clause_lstm_batch = tf.stack(values=splitted_init_clauses, axis=1)
            #     init_clause_lstm_batch = tf.reshape(tensor=init_clause_lstm_batch,
            #                                         shape=[self.num_init_clauses, self.num_proof * self.num_shuffles,
            #                                                self.comb_features])
            #     # Split over time
            #     time_batches = tf.unstack(value=init_clause_lstm_batch, axis=0)
            #
            #     # Create first LSTM and initialize states
            #     self.lstm_one = tf.contrib.rnn.BasicLSTMCell(self.comb_features)
            #     hidden_state_one = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
            #     current_state_one = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
            #     self.state_lstm_one = hidden_state_one, current_state_one
            #
            #     # Run LSTM over all time steps and save output
            #     first_lstm_output = []
            #     all_states = []
            #     for batch in time_batches:
            #         output, self.state_lstm_one = self.lstm_one(batch, self.tuple_to_lstm_state(self.state_lstm_one))
            #         first_lstm_output.append(output)
            #         all_states.append(self.state_lstm_one)
            #     self.state_lstm_one = self.extract_states(all_states)

            with tf.variable_scope("LSTM_INITIAL"):
                with tf.name_scope("Preparation"):
                    # Prepare batch for LSTM. New shape: [Time, Proofs/Clauses, Clause-Features]
                    init_clause_lstm_batch = tf.stack(values=splitted_init_clauses, axis=1)
                    init_clause_lstm_batch = tf.reshape(tensor=init_clause_lstm_batch,
                                                        shape=[self.num_init_clauses, self.num_proof * self.num_shuffles,
                                                               self.comb_features])
                    # Split over time
                    time_batches = tf.unstack(value=init_clause_lstm_batch, axis=0)

                    self.lstm_initial = tf.contrib.rnn.BasicLSTMCell(self.comb_features)
                    hidden_state_initial = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                    self.state_lstm_initial = hidden_state_initial, hidden_state_initial

                with tf.name_scope("LSTM"):
                    # Run second LSTM over all time steps. Output will not be used anymore
                    all_states = []
                    for batch in time_batches:
                        _, self.state_lstm_initial = self.lstm_initial(batch,
                                                                       self.tuple_to_lstm_state(self.state_lstm_initial))
                        all_states.append(self.state_lstm_initial)

                    self.state_lstm_initial = self.extract_states(all_states)
                    # lstm_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)
                    # self.state_lstm_initial = tf.Print(self.state_lstm_initial,
                    #                                    [tf.reduce_max(lstm_var) for lstm_var in lstm_variables],
                    #                                    message="State Variables:",
                    #                                    summarize=8)

    def extract_states(self, all_states):
        with tf.name_scope("ExtractStates"):
            # Extract these states that have only seen real initial clauses and no default ones
            chosen_states_h = self.short_state_extraction(all_states, state_index=0)
            chosen_states_c = self.short_state_extraction(all_states, state_index=1)
            return [tf.stack(values=chosen_states_h, axis=0), tf.stack(values=chosen_states_c, axis=0)]

    def tuple_to_lstm_state(self, state_tuple):
        with tf.name_scope("StateConversion"):
            return state_tuple[0], tf.nn.tanh(state_tuple[0]) # tf.stop_gradient(tf.nn.tanh(state_tuple[0]))

    def short_state_extraction(self, all_states, state_index):
        state_tensor = tf.stack(values=[a[state_index] for a in all_states], axis=0)
        return [state_tensor[self.init_clauses_length[i_proof] - 1, i_proof * self.num_shuffles + i_shuffle, :]
                for i_proof in range(self.num_proof) for i_shuffle in range(self.num_shuffles)]

    def first_combination_layer(self, clause_vector, neg_conj_vector, reuse=False):
        global FC_LAYER_1
        concatenated_features = concat(
            values=[clause_vector, neg_conj_vector], axis=1,
            name="ConcatClauseNegConj")

        comb_layer = fully_connected(concatenated_features, self.comb_features, activation_fn=tf.nn.relu, reuse=reuse,
                                     name=FC_LAYER_1, use_batch_norm=False)
        return tf.squeeze(comb_layer)

    def embed_neg_conjecture(self):
        self.neg_conjecture_embedder = CNNEmbedder(embedding_size=self.embedding_size, name="NegConjectureEmbedder",
                                                   reuse_vocab=True, batch_size=self.num_proof, char_number=None,
                                                   net_type=self.embedding_net_type, tensor_height=1,
                                                   use_batch_norm=False, wavenet_blocks=self.wavenet_blocks,
                                                   wavenet_layers=self.wavenet_layers, dropout_rate=self.dropout_rate_embedder,
                                                   is_training=self.is_training, use_conversion=self.use_conversion)
        # Squeeze so that LSTMs can run with fully connected layers
        self.neg_conj_embedded = tf.squeeze(self.neg_conjecture_embedder.embedded_vector)

    # [a b c d] to [a b a b c d c d]
    def repeat_initial_state(self, input_tensor):
        input_tensor = tf.reshape(input_tensor, [self.num_proof, self.num_shuffles, -1])
        input_tensor = tf.tile(input_tensor, multiples=[1, int(self.num_train_clauses / self.num_shuffles), 1])
        input_tensor = tf.reshape(input_tensor, [int(self.num_proof * self.num_train_clauses), -1])
        return input_tensor

    @staticmethod
    def create_shuffle_tensor(num_init_clauses, num_shuffles):
        all_shuffles = list()
        min_fact = -1
        for shuff_index in range(num_shuffles):
            shuffle_matrix = [list(range(num_init_clauses)) for _ in range(num_init_clauses)]
            for i in range(num_init_clauses):
                a = shuffle_matrix[i][:i + 1]
                if math.factorial(i+1) <= num_shuffles:
                    permuts = list(itertools.permutations(range(0,i+1)))
                    a = permuts[shuff_index % len(permuts)]
                else:
                    if min_fact == -1:
                        min_fact = i
                    for _ in range(100):
                        shuffle(a)
                        already_exists = False
                        for j in range(shuff_index):
                            diff_list = [k for k in range(i+1) if a[k] != all_shuffles[j][i][k]]
                            # print(str(a) + " vs " + str(all_shuffles[j][i][:i+1]) + " -> " + str(diff_list))
                            already_exists = already_exists or (len(diff_list) < (i - min_fact + 1))
                        if not already_exists:
                            break
                shuffle_matrix[i][:i + 1] = a
            all_shuffles.append(shuffle_matrix)
        return all_shuffles

    # Convert [a b c d] to [a a b b c c d d]
    @staticmethod
    def repeat_tensor(tensor_to_repeat, times, axis, name=None):
        if name is None:
            name = "RepeatTensorAx" + str(axis) + "x" + str(times)
        with tf.name_scope(name):
            input_shape = tensor_to_repeat.get_shape().as_list()
            # print(input_shape)
            output_shape = input_shape[:]
            output_shape[axis] *= times
            multiples = [1 for _ in range(len(input_shape))]
            multiples.insert(axis + 1, int(times))
            input_shape.insert(axis + 1, 1)
            tensor_to_repeat = tf.reshape(tensor_to_repeat, shape=input_shape)
            tensor_to_repeat = tf.tile(tensor_to_repeat, multiples=multiples)
            tensor_to_repeat = tf.reshape(tensor_to_repeat, shape=output_shape)
            return tensor_to_repeat


def test_repeat():
    a = tf.constant([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    b = CombLSTMNetwork.repeat_tensor(a, 5, 1)
    with tf.Session() as sess:
        initialize_tf_variables()
        out = sess.run([b], feed_dict={})
        print(out)


def test_extract_states():
    num_proofs = 4
    num_shuffles = 4
    num_init_clauses = 4
    overall_states = []
    for l in range(num_init_clauses):
        batch = []
        for i in range(num_proofs):
            for j in range(num_shuffles):
                batch.append(list(range(i * num_shuffles + j + 100 * l, i * num_shuffles + j + 4 + 100 * l)))
        a = [tf.constant(batch)]
        overall_states.append(a)
    print(overall_states)
    with tf.Session() as sess:
        network = CombLSTMNetwork(num_init_clauses=num_init_clauses, num_shuffles=num_shuffles, num_proof=num_proofs,
                                  num_train_clauses=num_init_clauses)
        initialize_tf_variables()
        b = network.short_state_extraction(overall_states, 0)
        out = sess.run(b, feed_dict={network.init_clauses_length: [4, 3, 3, 1]})
        print(out)


def test_shuffle_tensor():
    num_init_clauses = 8
    num_shuffles = 8
    shuffle_matrix = CombLSTMNetwork.create_shuffle_tensor(num_init_clauses, num_shuffles)
    for i in range(num_init_clauses):
        print("[")
        for j in range(num_shuffles):
            print(shuffle_matrix[j][i])
        print("]")


def visualize_graph():
    with tf.Session() as sess:
        network = CombLSTMNetwork()
        initialize_tf_variables()
        writer = create_summary_writer('logs/CombLSTMNet/', sess=sess)


if __name__ == '__main__':
    # test_extract_states()
    # visualize_graph()
    test_shuffle_tensor()
