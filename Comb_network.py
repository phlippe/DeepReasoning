from ops import *


class CombNetwork:
    def __init__(self, clause_embedder, neg_conjecture_embedder, comb_features=1024, name="CombNet", use_neg_conj=True,
                 weight0=1, weight1=1):

        assert comb_features > 0, "Number of channels for first layer has to be greater than 0!"
        assert clause_embedder is not None, "Clause embedder can not be None!"
        assert neg_conjecture_embedder is not None, "Negated conjecture embedder can not be None!"

        self.clause_embedder = clause_embedder
        self.neg_conjecture_embedder = neg_conjecture_embedder
        self.comb_features = comb_features
        self.name = name
        self.useNegConj = use_neg_conj
        self.tensor_height = clause_embedder.tensor_height
        self.weight0 = weight0
        self.weight1 = weight1
        self.labels = None
        self.weight = None
        self.loss = None
        self.loss_ones = None
        self.loss_zeros = None
        self.forward()

    def forward(self):
        with tf.variable_scope(self.name):
            self.labels = tf.placeholder(dtype="float32", shape=[self.clause_embedder.batch_size], name="Labels")
            if self.useNegConj:
                neg_embedded = self.neg_conjecture_embedder.embedded_vector
            else:
                neg_embedded = self.neg_conjecture_embedder
            concatenated_features = concat(
                values=[self.clause_embedder.embedded_vector, neg_embedded], axis=3,
                name="Concat")
            layer1 = fully_connected(concatenated_features, self.comb_features, activation_fn=tf.nn.relu, reuse=False,
                                     name="Comb_1024", use_batch_norm=False)
            self.weight = fully_connected(layer1, 1, activation_fn=tf.nn.sigmoid, reuse=False, name="Comb_final", use_batch_norm=False)
            self.weight = tf.squeeze(self.weight, name="CalcWeights")
            if self.tensor_height != 1:
                self.weight = tf.reshape(tensor=self.weight, shape=[-1], name="ReshapeTo1D")
            self.loss, self.loss_ones, self.loss_zeros, _ = weighted_BCE_loss(self.weight, self.labels, self.weight0,
                                                                              self.weight1)
