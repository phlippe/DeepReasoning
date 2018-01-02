from model_trainer import ModelTrainer
from data_loader import ClauseLoader
from CNN_embedder_network import CNNEmbedder
from Comb_network import CombNetwork


class CNNEmbedderTrainer(ModelTrainer):

    def __init__(self, train_files, test_files, use_wavenet=False):
        self.train_loader = ClauseLoader(file_list=train_files, prob_pos=0.5)
        self.test_loader = ClauseLoader(file_list=test_files, augment=False)
        self.train_loader.print_statistic()
        self.test_loader.print_statistic()

        self.use_wavenet = use_wavenet

    def create_model(self, batch_size, embedding_size):
        clause_embedder = CNNEmbedder(embedding_size=embedding_size, name="ClauseEmbedder",
                                      batch_size=batch_size, char_number=None, use_wavenet=self.use_wavenet)
        neg_conjecture_embedder = CNNEmbedder(embedding_size=embedding_size, name="NegConjectureEmbedder",
                                              reuse_vocab=True, batch_size=batch_size, char_number=None,
                                              use_wavenet=self.use_wavenet)
        combined_network = CombNetwork(clause_embedder, neg_conjecture_embedder, weight0=1, weight1=1)
        return combined_network

    def run_model(self, sess, model, fetches, batch):
        input_clause, input_clause_len, input_conj, input_conj_len, labels = batch
        feed_dict = {
            model.clause_embedder.input_clause: input_clause,
            model.neg_conjecture_embedder.input_clause: input_conj,
            model.clause_embedder.input_length: input_clause_len,
            model.neg_conjecture_embedder.input_length: input_conj_len,
            model.labels: labels
        }
        return sess.run(fetches, feed_dict=feed_dict)

    def get_train_batch(self, batch_size):
        return self.train_loader.get_batch(batch_size)

    def get_test_batch(self, batch_size):
        return self.test_loader.get_batch(batch_size)

    def process_specific_loss_information(self, all_losses):
        pass