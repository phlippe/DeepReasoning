from model_trainer import ModelTrainer
from initial_data_loader import InitialClauseLoader
from Comb_LSTM_network import CombLSTMNetwork

import numpy as np


class CombLSTMTrainer(ModelTrainer):
    def __init__(self, train_files, test_files, num_proofs, num_initial_clauses, num_training_clauses, num_shuffles,
                 val_batch_number, prob_pos=0.3):
        self.val_batch_number = val_batch_number
        self.val_batches = []
        self.val_index = 0
        self.num_proofs = num_proofs
        self.num_initial_clauses = num_initial_clauses
        self.num_training_clauses = num_training_clauses
        self.num_shuffles = num_shuffles
        self.batch_files = []

        self.prob_pos = prob_pos
        self.train_loader = InitialClauseLoader(file_list=train_files, prob_pos=self.prob_pos)
        self.test_loader = InitialClauseLoader(file_list=test_files, prob_pos=0.5)
        self.train_loader.print_statistic()
        self.test_loader.print_statistic()
        for i in range(self.val_batch_number):
            self.val_batches.append(self.test_loader.get_batch(num_proofs=self.num_proofs,
                                                               num_training_clauses=self.num_training_clauses,
                                                               num_init_clauses=self.num_initial_clauses))

    def create_model(self, batch_size, embedding_size):
        combined_network = CombLSTMNetwork(num_proof=self.num_proofs, num_train_clauses=self.num_training_clauses,
                                           num_shuffles=self.num_shuffles, num_init_clauses=self.num_initial_clauses,
                                           weight0=0.5/self.prob_pos, weight1=0.5/(1-self.prob_pos))
        return combined_network

    def run_model(self, sess, model, fetches, batch):
        input_clause, input_clause_len, input_conj, input_conj_len, init_clause_len, labels = batch
        feed_dict = {
            model.clause_embedder.input_clause: input_clause,
            model.neg_conjecture_embedder.input_clause: input_conj,
            model.clause_embedder.input_length: input_clause_len,
            model.neg_conjecture_embedder.input_length: input_conj_len,
            model.init_clauses_length: init_clause_len,
            model.labels: labels
        }
        return sess.run(fetches, feed_dict=feed_dict)

    def get_train_batch(self, batch_size):
        batch = self.train_loader.get_batch(num_proofs=self.num_proofs, num_training_clauses=self.num_training_clauses,
                                            num_init_clauses=self.num_initial_clauses)
        self.batch_files = batch[-1]
        return batch[:-1]

    def get_test_batch(self, batch_size):
        if self.val_index >= len(self.val_batches) or self.val_index < 0:
            self.val_index = 0
        batch = self.val_batches[self.val_index]
        self.val_index += 1
        return batch

    def get_batch_size(self):
        return self.num_proofs * (self.num_initial_clauses + self.num_training_clauses)

    def print_specific_loss_information(self, all_losses):
        s = "\t"*4
        proof_size = len(all_losses) / self.num_proofs
        for i in range(self.num_proofs):
            if i > 0:
                s += ", "
            proof_loss = np.mean(all_losses[i*proof_size:(i+1)*proof_size])
            s += self.batch_files[i]+": "+str(proof_loss)
        print(s)

