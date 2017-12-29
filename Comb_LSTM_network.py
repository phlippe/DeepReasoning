from ops import *
from CNN_embedder_network import CNNEmbedder


class CombLSTMNetwork:
    FC_LAYER_1 = "Comb_1024"
    FC_LAYER_2 = "Comb_final"

    def __init__(self, comb_features=1024, num_init_clauses=32, num_proof=4, num_train_clauses=32, num_shuffles=4,
                 weight0=1, weight1=1, name="CombLSTMNet"):

        assert comb_features > 0, "Number of channels for first layer has to be greater than 0!"

        self.num_init_clauses = num_init_clauses
        self.num_proof = num_proof
        self.num_train_clauses = num_train_clauses
        self.batch_size = num_train_clauses * num_proof
        self.num_shuffles = num_shuffles
        self.comb_features = comb_features
        self.name = name
        self.tensor_height = 1
        self.weight0 = weight0
        self.weight1 = weight1
        self.labels = None
        self.weight = None
        self.loss = None
        self.loss_ones = None
        self.loss_zeros = None
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
        global FC_LAYER_1, FC_LAYER_2
        with tf.variable_scope(self.name):
            self.labels = tf.placeholder(dtype="float32", shape=[self.clause_embedder.batch_size], name="Labels")

            self.embed_clauses()
            self.embed_neg_conjecture()

            self.run_init_clauses()
            self.run_comb_layers()

    def run_comb_layers(self):
        with tf.name_scope("CombNetwork"):
            lstm_one_output, temp_state = self.lstm_one(self.train_clauses,
                                                        self.repeat_LSTM_states(self.state_lstm_one))

            neg_conj_vector = self.repeat_tensor(tensor_to_repeat=self.neg_conj_embedded, axis=0,
                                                 times=self.num_train_clauses / self.num_proof)

            layer1 = self.first_combination_layer(lstm_one_output, neg_conj_vector, reuse=True)

            lstm_two_output, temp_state = self.lstm_two(layer1, self.repeat_LSTM_states(self.state_lstm_two))

            self.weight = fully_connected(input_=lstm_two_output, outputs=1, activation_fn=tf.nn.sigmoid, reuse=False,
                                          name=FC_LAYER_2, use_batch_norm=False)
            self.weight = tf.squeeze(self.weight, name="CalcWeights")
            if self.tensor_height != 1:
                self.weight = tf.reshape(tensor=self.weight, shape=[-1], name="ReshapeTo1D")
            self.loss, self.loss_ones, self.loss_zeros = weighted_BCE_loss(self.weight, self.labels, self.weight0,
                                                                           self.weight1)

    def repeat_LSTM_states(self, state):
        rep_factor = self.num_train_clauses / self.num_proof / self.num_shuffles
        repeated_hidden_states = self.repeat_tensor(tensor_to_repeat=state[0], axis=0, times=rep_factor)
        repeated_current_states = self.repeat_tensor(tensor_to_repeat=state[1], axis=0, times=rep_factor)
        return [repeated_hidden_states, repeated_current_states]

    def embed_clauses(self):
        self.clause_embedder = CNNEmbedder(embedding_size=1024, name="ClauseEmbedder",
                                           batch_size=self.batch_size + self.num_init_clauses * self.num_proof,
                                           char_number=None, use_wavenet=False, tensor_height=1)
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
                # Randomly shuffle init clauses. For better generalization do it "self.num_shuffles" times differently.
                splitted_init_clauses = [
                    [tf.random_shuffle(value=splitted_init_clauses[i][:self.init_clauses_length[i]]) for _ in
                     range(self.num_shuffles)] for i in
                    range(len(splitted_init_clauses))]  # Shuffle init clauses for better generalization
                splitted_init_clauses = [tensor for sublist in splitted_init_clauses for tensor in
                                         sublist]  # Flatten list

            with tf.name_scope("LSTM_ONE"):
                # Prepare batch for first LSTM. New shape: [Time, Proofs/Clauses, Clause-Features]
                init_clause_lstm_batch = tf.concat(values=splitted_init_clauses, axis=1)
                # Split over time
                time_batches = tf.split(value=init_clause_lstm_batch, axis=0)

                # Create first LSTM and initialize states
                self.lstm_one = tf.contrib.rnn.BasicLSTMCell(self.comb_features)
                hidden_state_one = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                current_state_one = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                self.state_lstm_one = hidden_state_one, current_state_one

                # Run LSTM over all time steps and save output
                first_lstm_output = []
                for batch in time_batches:
                    output, self.state_lstm_one = self.lstm_one(batch, self.state_lstm_one)
                    first_lstm_output.append(output)

            with tf.name_scope("VectorPreparation"):
                # Use output of first LSTM to determine the states of the second one
                # Run through first fully connected layer
                clause_vector = tf.concat(values=first_lstm_output, axis=0)
                neg_conj_vector = self.repeat_tensor(tensor_to_repeat=self.neg_conj_embedded, times=self.num_shuffles,
                                                     axis=0)
                neg_conj_vector = tf.tile(neg_conj_vector, multiples=self.init_clauses_length)

            comb_layer = self.first_combination_layer(clause_vector, neg_conj_vector, reuse=False)

            with tf.name_scope("LSTM_TWO"):
                time_batches = tf.split(value=comb_layer, axis=0, num_or_size_splits=self.num_init_clauses)

                self.lstm_two = tf.contrib.rnn.BasicLSTMCell(self.comb_features)
                hidden_state_two = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                current_state_two = tf.zeros(shape=[self.num_proof * self.num_shuffles, self.comb_features])
                self.state_lstm_two = hidden_state_two, current_state_two

                # Run second LSTM over all time steps. Output will not be used anymore
                for batch in time_batches:
                    output, self.state_lstm_two = self.lstm_two(batch, self.state_lstm_two)

    # Convert [a b c d] to [a a b b c c d d]
    def repeat_tensor(self, tensor_to_repeat, times, axis, name=None):
        if name is None:
            name = "RepeatTensorAx" + str(axis) + "x" + str(times)
        with tf.name_scope(name):
            input_shape = tensor_to_repeat.get_shape()
            output_shape = input_shape.copy()
            output_shape[axis] += times
            multiples = [1 for _ in range(len(input_shape))]
            multiples.insert(index=axis + 1, object=times)
            input_shape.insert(index=axis + 1, object=1)
            tensor_to_repeat = tf.reshape(tensor_to_repeat, shape=input_shape)
            tensor_to_repeat = tf.tile(tensor_to_repeat, multiples=multiples)
            tensor_to_repeat = tf.reshape(tensor_to_repeat, shape=output_shape)
            return tensor_to_repeat

    def first_combination_layer(self, clause_vector, neg_conj_vector, reuse=False):
        concatenated_features = concat(
            values=[clause_vector, neg_conj_vector], axis=1,
            name="ConcatClauseNegConj")

        comb_layer = fully_connected(concatenated_features, self.comb_features, activation_fn=tf.nn.relu, reuse=reuse,
                                     name=FC_LAYER_1, use_batch_norm=False)
        return tf.squeeze(comb_layer)

    def embed_neg_conjecture(self):
        self.neg_conjecture_embedder = CNNEmbedder(embedding_size=1024, name="NegConjectureEmbedder",
                                                   reuse_vocab=True, batch_size=self.num_proof, char_number=None,
                                                   use_wavenet=False, tensor_height=1)
        # Squeeze so that LSTMs can run with fully connected layers
        self.neg_conj_embedded = tf.squeeze(self.neg_conjecture_embedder.embedded_vector)
