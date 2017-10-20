from ops import *


class WaveNet:
    def __init__(self, layer_number, clause_number, embedding_size, kernel_size=3, p=0.2, block_number=1, batch_size=1,
                 channel_size=-1):
        """
        WaveNet for automated theorem proving
        :param layer_number: Number of layers per WaveNet-Block
        :param clause_number: Number of clauses which are input to the network
        :param embedding_size: Channel size of the embedded clauses
        :param kernel_size: Size of kernels for dilated convolutions in the WaveNet-Blocks
        :param p: Dropout probability for every dropout layer
        :param block_number: Number of stacked WaveNet-Blocks
        :param batch_size: Batch size for training
        :param channel_size: Inner channel size for dilated convolutions
        """
        assert layer_number > 0, "Number of layers can not be negative nor 0"
        assert block_number > 0, "Number of blocks can not be negative nor 0"
        assert clause_number > 0, "Number of clauses can not be negative nor 0"
        assert embedding_size > 0, "The embedding size can not be negative nor 0"
        assert kernel_size > 0, "The kernel size can not be negative nor 0"
        assert p >= 0.0, "The dropout rate has to be positive"
        assert 1 > p, "The dropout rate has to be smaller 1"
        assert batch_size > 0, "The batch size can not be negative nor 0"

        self.layer_number = layer_number
        self.clause_number = clause_number
        self.embedding_size = embedding_size
        self.kernel_size = kernel_size
        self.dropout_p = p
        self.block_number = block_number
        self.batch_size = batch_size
        self.channel_size = channel_size
        if self.channel_size <= 0:
            self.channel_size = self.embedding_size

        self.clause_tensor = None
        self.negated_conjecture = None
        self.final_values = None
        self.labels = None
        self.loss = None

        self.forward()

    def forward(self):
        """

        :return:
        """
        self.clause_tensor = tf.placeholder(dtype="float32",
                                            shape=[self.batch_size, 1, self.clause_number, self.embedding_size],
                                            name="Input_Clauses")

        self.negated_conjecture = tf.placeholder(dtype="float32",
                                                 shape=[self.batch_size, self.embedding_size],
                                                 name="Negated_Conjecture")

        combined_clauses = self.combiner_network(self.clause_tensor, self.negated_conjecture, reuse=False)

        compared_clauses = self.hierarchical_wavenet(combined_clauses, reuse=False)

        self.final_values = self.compresser_network(compared_clauses, reuse=False)

        self.loss = self.loss_calculation(self.final_values, self.labels)

    def combiner_network(self, clauses, negated_conjecture, last_iteration=None, reuse=False):
        negated_conjecture = tf.reshape(negated_conjecture, shape=[self.batch_size, 1, 1, self.embedding_size])
        negated_conjecture = tf.tile(negated_conjecture, multiples=[1, 1, self.clause_number, 1])

        concatenated_clauses = tf.concat([clauses, negated_conjecture], axis=3, name="Clause_Concatenation")
        if last_iteration is not None:
            concatenated_clauses = tf.concat([concatenated_clauses, last_iteration], axis=3, name="Recursive_Concat")

        return conv1d(concatenated_clauses, output_dim=self.channel_size, kernel_size=1, name="CombNet_1x1_1",
                      reuse=reuse)

    def hierarchical_wavenet(self, input_tensor, reuse=False):
        for block_index in xrange(self.block_number):
            """
            B(x) = x + (L_{64} * L_{32} * ... * L_{1})(D_{f}(x,p))
            """
            if block_index == 0:
                block_tensor = dropout(input_tensor, self.dropout_p)
            else:
                block_tensor = dropout(block_tensor, self.dropout_p)
            layer_tensor = block_tensor

            for layer_index in xrange(7):
                layer_tensor = wavenet_layer(input_=layer_tensor, kernel_size=self.kernel_size,
                                             dilation_rate=2 ** layer_index,
                                             name="WaveNetLayer" + str(layer_index), reuse=reuse)
            block_tensor = block_tensor + layer_tensor
        return block_tensor

    def compresser_network(self, compared_clauses, reuse=False):
        return conv1d(compared_clauses, output_dim=1, kernel_size=1, name="CombNet_1x1_1",
                      reuse=reuse)

    def loss_calculation(self, input_tensor, labels):
        return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=input_tensor)
