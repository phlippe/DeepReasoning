from ops import *
from CNN_embedder_network import CNNEmbedder, NetType
from random import shuffle

FC_LAYER_1 = "Comb_1024"
FC_LAYER_2 = "Comb_final"


class CombLSTMNetwork:
    def __init__(self, embedding_size=1024, num_init_clauses=32, num_proof=4, num_train_clauses=32, num_shuffles=4,
                 weight0=1, weight1=1, name="CombLSTMNet", wavenet_blocks=1, wavenet_layers=2, comb_features=1024,
                 embedding_net_type=NetType.STANDARD):

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
        self.labels = None
        self.weight = None
        self.loss = None
        self.loss_ones = None
        self.loss_zeros = None
        self.all_losses = None
        self.init_clauses = None
        self.train_clauses = None
        self.clause_embedder = None
        self.neg_conjecture_embedder = None
        self.neg_conj_embedded = None
        self.init_clauses_length = None
        self.lstm_one = None
        self.lstm_two = None
        self.state_lstm_one = None
        self.state_lstm_two = None
        self.forward()

    def forward(self):
        with tf.variable_scope(self.name):
            self.labels = tf.placeholder(dtype="float32", shape=[self.batch_size], name="Labels")

            self.embed_clauses()
            self.embed_neg_conjecture()

            self.run_init_clauses()
            self.run_comb_layers()

    def run_comb_layers(self):
        global FC_LAYER_2
        with tf.name_scope("CombNetwork"):
            lstm_one_output, temp_state = self.lstm_one(self.train_clauses,
                                                        self.repeat_lstm_states(self.state_lstm_one))

            neg_conj_vector = self.repeat_tensor(tensor_to_repeat=self.neg_conj_embedded, axis=0,
                                                 times=self.num_train_clauses)

            layer1 = self.first_combination_layer(lstm_one_output, neg_conj_vector, reuse=True)

            lstm_two_output, temp_state = self.lstm_two(layer1, self.repeat_lstm_states(self.state_lstm_two))

            self.weight = fully_connected(input_=lstm_two_output, outputs=1, activation_fn=tf.nn.sigmoid, reuse=False,
                                          name=FC_LAYER_2, use_batch_norm=False)
            self.weight = tf.squeeze(self.weight, name="CalcWeights")
            if self.tensor_height != 1:
                self.weight = tf.reshape(tensor=self.weight, shape=[-1], name="ReshapeTo1D")
            self.loss, self.loss_ones, self.loss_zeros, self.all_losses = weighted_BCE_loss(self.weight, self.labels,
                                                                                            self.weight0, self.weight1)

    def repeat_lstm_states(self, state):
        rep_factor = int(self.num_train_clauses / self.num_shuffles)
        print("Repeat factor: " + str(rep_factor))
        print("Hidden states: " + str(state[0].get_shape().as_list()))
        print("Current states: " + str(state[1].get_shape().as_list()))
        repeated_hidden_states = CombLSTMNetwork.repeat_tensor(tensor_to_repeat=state[0], axis=0, times=rep_factor)
        repeated_current_states = CombLSTMNetwork.repeat_tensor(tensor_to_repeat=state[1], axis=0, times=rep_factor)
        print("Repeated hidden states: " + str(repeated_hidden_states.get_shape().as_list()))
        print("Repeated current states: " + str(repeated_current_states.get_shape().as_list()))
        return [repeated_hidden_states, repeated_current_states]

    def embed_clauses(self):
        self.clause_embedder = CNNEmbedder(embedding_size=self.embedding_size, name="ClauseEmbedder",
                                           batch_size=self.batch_size + self.num_init_clauses * self.num_proof,
                                           char_number=None, net_type=self.embedding_net_type, tensor_height=1,
                                           use_batch_norm=False, wavenet_blocks=self.wavenet_blocks,
                                           wavenet_layers=self.wavenet_layers)
        # Squeeze so that LSTMs can run with fully connected layers
        all_clauses = tf.split(tf.squeeze(self.clause_embedder.embedded_vector), num_or_size_splits=2, axis=0)
        self.train_clauses = all_clauses[0]
        self.init_clauses = all_clauses[1]

    def run_init_clauses(self):
        # Number of real init clauses that are given. Maximum is "self.num_init_clauses".
        # Other places must be filled up with default/random clause
        with tf.name_scope("InitialClauses"):
            self.init_clauses_length = tf.placeholder(dtype="int32", shape=[self.num_proof], name="InitClausesLength")

            with tf.name_scope("ShuffleInitClauses"):
                # Split clauses belonging to different proofs/negated conjectures
                splitted_init_clauses = tf.split(value=self.init_clauses, num_or_size_splits=self.num_proof, axis=0)
                # Create randomly shuffled index matrices for all init clause lengths
                shuffle_list = [CombLSTMNetwork.create_shuffle_tensor(self.num_init_clauses) for _ in
                                range(self.num_shuffles)]
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

            with tf.variable_scope("LSTM_ONE"):
                # Prepare batch for LSTM. New shape: [Time, Proofs/Clauses, Clause-Features]
                init_clause_lstm_batch = tf.stack(values=splitted_init_clauses, axis=1)
                init_clause_lstm_batch = tf.reshape(tensor=init_clause_lstm_batch,
                                                    shape=[self.num_init_clauses, self.num_proof * self.num_shuffles,
                                                           self.comb_features])
                # Split over time
                time_batches = tf.unstack(value=init_clause_lstm_batch, axis=0)

                # Create first LSTM and initialize states
                self.lstm_one = tf.contrib.rnn.BasicLSTMCell(self.comb_features)
                hidden_state_one = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                current_state_one = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                self.state_lstm_one = hidden_state_one, current_state_one

                # Run LSTM over all time steps and save output
                first_lstm_output = []
                all_states = []
                for batch in time_batches:
                    output, self.state_lstm_one = self.lstm_one(batch, self.state_lstm_one)
                    first_lstm_output.append(output)
                    all_states.append(self.state_lstm_one)
                self.state_lstm_one = self.extract_states(all_states)

            with tf.name_scope("VectorPreparation"):
                # Use output of first LSTM to determine the states of the second one
                # Run through first fully connected layer
                clause_vector = tf.concat(values=first_lstm_output, axis=0)
                neg_conj_vector = self.repeat_tensor(tensor_to_repeat=self.neg_conj_embedded, times=self.num_shuffles,
                                                     axis=0)
                neg_conj_vector = tf.tile(neg_conj_vector, multiples=[self.num_init_clauses, 1])
                # print("Clause vector: " + str(clause_vector.get_shape().as_list()))
                # print("Neg Conj vector: " + str(neg_conj_vector.get_shape().as_list()))

            comb_layer = self.first_combination_layer(clause_vector, neg_conj_vector, reuse=False)

            with tf.variable_scope("LSTM_TWO"):
                time_batches = tf.split(value=comb_layer, axis=0, num_or_size_splits=self.num_init_clauses)

                self.lstm_two = tf.contrib.rnn.BasicLSTMCell(self.comb_features)
                hidden_state_two = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                current_state_two = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                self.state_lstm_two = hidden_state_two, current_state_two

                # Run second LSTM over all time steps. Output will not be used anymore
                all_states = []
                for batch in time_batches:
                    output, self.state_lstm_two = self.lstm_two(batch, self.state_lstm_two)
                    all_states.append(self.state_lstm_two)
                self.state_lstm_two = self.extract_states(all_states)

    def extract_states(self, all_states):
        with tf.name_scope("ExtractStates"):
            # Extract these states that have only seen real initial clauses and no default ones
            chosen_states_h = self.short_state_extraction(all_states, state_index=0)
            chosen_states_c = self.short_state_extraction(all_states, state_index=1)
            return [tf.stack(values=chosen_states_h, axis=0), tf.stack(values=chosen_states_c, axis=0)]

    def short_state_extraction(self, all_states, state_index):
        # print("All states size: " + str(len([a[state_index] for a in all_states])))
        state_tensor = tf.stack(values=[a[state_index] for a in all_states], axis=0)
        # print("State tensor size: " + str(state_tensor.get_shape().as_list()))
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
                                                   wavenet_layers=self.wavenet_layers)
        # Squeeze so that LSTMs can run with fully connected layers
        self.neg_conj_embedded = tf.squeeze(self.neg_conjecture_embedder.embedded_vector)

    @staticmethod
    def create_shuffle_tensor(num_init_clauses):
        shuffle_matrix = [list(range(num_init_clauses)) for _ in range(num_init_clauses)]
        for i in range(num_init_clauses):
            a = shuffle_matrix[i][:i + 1]
            shuffle(a)
            shuffle_matrix[i][:i + 1] = a
        return shuffle_matrix

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
    num_init_clauses = 32
    shuffle_matrix = CombLSTMNetwork.create_shuffle_tensor(num_init_clauses)
    print(shuffle_matrix)


def visualize_graph():
    with tf.Session() as sess:
        network = CombLSTMNetwork()
        initialize_tf_variables()
        writer = create_summary_writer('logs/CombLSTMNet/', sess=sess)


if __name__ == '__main__':
    # test_extract_states()
    # visualize_graph()
    test_shuffle_tensor()
