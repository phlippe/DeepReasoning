from model_trainer import ModelTrainer
from data_loader import ClauseLoader
from CNN_embedder_network import CNNEmbedder
from Comb_network import CombNetwork


class CNNEmbedderTrainer(ModelTrainer):

    def __init__(self, train_files, test_files, use_wavenet=False):
        self.train_loader = ClauseLoader(file_list=train_files, prob_pos=0.5)
        self.val_loader = ClauseLoader(file_list=test_files, augment=False)
        self.train_loader.print_statistic()
        self.val_loader.print_statistic()

        self.use_wavenet = use_wavenet

    def create_model(self, batch_size, embedding_size):
        clause_embedder = CNNEmbedder(embedding_size=embedding_size, name="ClauseEmbedder",
                                      batch_size=batch_size, char_number=None)
        neg_conjecture_embedder = CNNEmbedder(embedding_size=embedding_size, name="NegConjectureEmbedder",
                                              reuse_vocab=True, batch_size=batch_size, char_number=None)
        combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder, weight0=1, weight1=1)
        return combined_network

    def run_model(self, sess, model, fetches, batch, is_training=True, run_options=None, run_metadata=None):
        input_clause, input_clause_len, input_conj, input_conj_len, labels = batch
        feed_dict = {
            model.clause_embedder.input_clause: input_clause,
            model.neg_conjecture_embedder.input_clause: input_conj,
            model.clause_embedder.input_length: input_clause_len,
            model.neg_conjecture_embedder.input_length: input_conj_len,
            model.labels: labels
        }
        return sess.run(fetches, feed_dict=feed_dict, run_options=run_options, run_metadata=run_metadata)

    def get_train_batch(self, batch_size):
        return self.train_loader.get_batch(batch_size)

    def get_val_batch(self, batch_size):
        return self.val_loader.get_batch(batch_size)

    def process_specific_loss_information(self, all_losses):
        pass

    def get_test_batches(self, batch_size):
        return []

    def process_test_batches(self, weights):
        pass
