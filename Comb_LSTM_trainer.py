from model_trainer import ModelTrainer
from initial_data_loader import InitialClauseLoader
from test_data_loader import TestClauseLoader
from Comb_LSTM_network import CombLSTMNetwork
from CNN_embedder_network import NetType

import numpy as np


class CombLSTMTrainer(ModelTrainer):
    def __init__(self, train_files, val_files, test_files, num_proofs, num_initial_clauses, num_training_clauses,
                 num_shuffles, val_batch_number, prob_pos=0.25, loss_filter_size=64, embedding_net_type=NetType.STANDARD,
                 wavenet_blocks=1, wavenet_layers=2, max_clause_len=150, max_neg_conj_len=150, use_conversion=False):
        self.val_batch_number = val_batch_number
        self.val_batches = []
        self.val_index = 0
        self.num_proofs = num_proofs
        self.num_initial_clauses = num_initial_clauses
        self.num_training_clauses = num_training_clauses
        self.num_shuffles = num_shuffles
        self.batch_proofs = []
        self.last_losses = []
        self.loss_filter_size = loss_filter_size
        self.wavenet_blocks = wavenet_blocks
        self.wavenet_layers = wavenet_layers
        self.embedding_net_type = embedding_net_type
        self.use_conversion = use_conversion

        self.prob_pos = prob_pos
        self.train_loader = InitialClauseLoader(file_list=train_files, prob_pos=self.prob_pos,
                                                max_clause_len=max_clause_len, max_neg_conj_len=max_neg_conj_len,
                                                use_conversion=use_conversion)
        self.val_loader = InitialClauseLoader(file_list=val_files, prob_pos=0.5, max_clause_len=max_clause_len,
                                              max_neg_conj_len=max_neg_conj_len, use_conversion=use_conversion)
        self.test_loader = TestClauseLoader(file_list=test_files, use_conversion=use_conversion)
        self.train_loader.print_statistic()
        self.val_loader.print_statistic()
        for i in range(self.val_batch_number):
            self.val_batches.append(self.val_loader.get_batch(num_proofs=self.num_proofs,
                                                              num_training_clauses=self.num_training_clauses,
                                                              num_init_clauses=self.num_initial_clauses)[:-1])
        self.test_batches = self.test_loader.get_all_batches(self.num_proofs, self.num_training_clauses,
                                                             self.num_initial_clauses)

    def create_model(self, batch_size, embedding_size):
        combined_network = CombLSTMNetwork(num_proof=self.num_proofs, num_train_clauses=self.num_training_clauses,
                                           num_shuffles=self.num_shuffles, num_init_clauses=self.num_initial_clauses,
                                           weight0=1, weight1=1.1, embedding_net_type=self.embedding_net_type,
                                           wavenet_blocks=self.wavenet_blocks, wavenet_layers=self.wavenet_layers,
                                           embedding_size=512, comb_features=1024, dropout_rate_embedder=0.2,
                                           dropout_rate_fc=0.1, use_conversion=self.use_conversion)
        return combined_network

    def run_model(self, sess, model, fetches, batch, is_training=True, run_options=None, run_metadata=None):
        input_clause, input_clause_len, input_clause_mask, input_clause_syntax, input_conj, input_conj_len, input_conj_mask, input_conj_syntax, init_clause_len, labels = batch

        feed_dict = {
            model.clause_embedder.input_clause: input_clause,
            model.neg_conjecture_embedder.input_clause: input_conj,
            model.clause_embedder.input_length: input_clause_len,
            model.neg_conjecture_embedder.input_length: input_conj_len,
            model.clause_embedder.input_mask: input_clause_mask,
            model.neg_conjecture_embedder.input_mask: input_conj_mask,
            model.clause_embedder.input_syntax: input_clause_syntax,
            model.neg_conjecture_embedder.input_syntax: input_conj_syntax,
            model.init_clauses_length: init_clause_len,
            model.labels: labels,
            model.is_training: is_training
        }
        # For timeline tracing
        return sess.run(fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

    def get_train_batch(self, batch_size):
        batch = self.train_loader.get_batch(num_proofs=self.num_proofs, num_training_clauses=self.num_training_clauses,
                                            num_init_clauses=self.num_initial_clauses)
        self.batch_proofs = batch[-1]
        return batch[:-1]

    def get_val_batch(self, batch_size):
        if self.val_index >= len(self.val_batches) or self.val_index < 0:
            self.val_index = 0
        batch = self.val_batches[self.val_index]
        self.val_index += 1
        return batch

    def get_batch_size(self):
        return self.num_proofs * (self.num_initial_clauses + self.num_training_clauses)

    def process_specific_loss_information(self, all_losses):
        s = "\t" * 6
        proof_size = int(len(all_losses) / self.num_proofs)
        all_proof_losses = []
        all_files = []
        for i in range(self.num_proofs):
            if i > 0:
                s += ", "
            proof_loss = np.mean(all_losses[i * proof_size:(i + 1) * proof_size])
            self.train_loader.proof_loader[self.batch_proofs[i]].loss_to_batch(all_losses[i * proof_size:(i + 1) * proof_size])
            all_proof_losses.append(proof_loss)
            file_name = self.train_loader.proof_loader[self.batch_proofs[i]].prefix.split("/", -1)[-1].split("_")[-1]
            all_files.append(file_name)
            s += file_name + ": " + str(proof_loss)
        print(s)

        if len(self.last_losses) > self.loss_filter_size - len(all_proof_losses):
            needed_spots = len(all_proof_losses) - (self.loss_filter_size - len(self.last_losses))
            if needed_spots > 0:
                del self.last_losses[:needed_spots]

        self.last_losses = self.last_losses + all_proof_losses
        loss_mean = np.mean(self.last_losses)

        for i in range(len(all_proof_losses)):
            if all_proof_losses[i]/loss_mean >= 1.5:
                self.train_loader.add_proof_index(self.batch_proofs[i])
                print(" [#] INFO: Adding extra index of "+all_files[i]+" to train loader ("+str(len(self.train_loader.proof_indices))+")")
            if all_proof_losses[i]/loss_mean <= 0.5:
                self.train_loader.remove_proof_index(self.batch_proofs[i])
                print(" [#] INFO: Removing an index of "+all_files[i]+" from train loader ("+str(len(self.train_loader.proof_indices))+")")

    def get_test_batches(self, batch_size):
        return [batch[:-1] for batch in self.test_batches]

    def process_test_batches(self, weights):
        return self.test_loader.print_out_results(self.test_batches, weights)

    def set_max_len(self, clause_len, conj_len):
        self.train_loader.set_max_len(clause_len, conj_len)
        self.val_loader.set_max_len(clause_len, conj_len)
