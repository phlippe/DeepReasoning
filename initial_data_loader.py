from data_loader import ProofExampleLoader, ClauseLoader, get_clause_lengths
from glob import glob
from data_augmenter import DataAugmenter, DefaultAugmenter

import math
from random import shuffle, randint, seed
import numpy as np
from TPTP_train_val_files import *

import sys
if sys.version_info[0] < 3:
    print(" [!] ERROR: Could not import module thread. Python version "+str(sys.version_info))
    sys.exit(1)
    # from thread import start_new_thread
else:
    from _thread import start_new_thread

LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1


class InitialClauseLoader:
    def __init__(self, file_list, empty_char=5, augment=True, prob_pos=0.3, index_divider=32, max_clause_len=150,
                 max_neg_conj_len=-1, use_conversion=False):
        if augment:
            self.augmenter = DataAugmenter(use_conversion=use_conversion)
        else:
            self.augmenter = DefaultAugmenter()
        self.empty_char = int(empty_char)
        self.prob_pos = prob_pos
        self.proof_loader = []
        self.proof_indices = []
        self.proof_index = 0
        self.pos_next = True
        self.global_batch = None
        self.max_clause_len = max_clause_len
        if max_neg_conj_len <= 0:
            self.max_neg_conj_len = self.max_clause_len
        else:
            self.max_neg_conj_len = max_neg_conj_len

        self.proof_loader = ClauseLoader.initialize_proof_loader(file_list, # [f for f in file_list if 'ClauseWeight_LCL' not in f],
                                                                 use_conversion=use_conversion)

        for index in range(len(self.proof_loader)):
            new_indices = [index for _ in range(int(math.ceil(
                (self.proof_loader[index].get_number_of_positives() +
                 math.sqrt(self.proof_loader[index].get_number_of_negatives())) * 1.0 / index_divider
            )))]
            self.proof_indices = self.proof_indices + new_indices

        seed()  # Initialize random number generator
        self.permute_proofs()

    def permute_proofs(self):
        shuffle(self.proof_indices)
        self.proof_index = 0

    def get_proof(self):
        if self.proof_index >= len(self.proof_indices):
            self.permute_proofs()
        proof = self.proof_indices[self.proof_index]
        self.proof_index += 1
        return proof

    def get_batch(self, num_proofs, num_training_clauses, num_init_clauses):
        if self.global_batch is None:
            self.__get_batch(num_proofs, num_training_clauses, num_init_clauses)
        current_batch = self.global_batch
        self.global_batch = None
        start_new_thread(self.__get_batch, (num_proofs, num_training_clauses, num_init_clauses))
        return current_batch

    def __get_batch(self, num_proofs, num_training_clauses, num_init_clauses):
        """
        Creating a batch of clauses with the following structure:
            batch = [training clauses, initial clauses]
            training clauses = num_proof * num_training_clauses * [pos/neg clauses of proof]
            initial clauses = num_proof * num_init_clauses * [initial clause of proof]

        :param num_proofs:
        :param num_training_clauses:
        :param num_init_clauses:
        :return:
        """
        global LABEL_NEGATIVE, LABEL_POSITIVE
        batch_neg_conj = []
        labels = np.zeros(shape=num_proofs*num_training_clauses, dtype=np.int32)
        training_clauses = []
        initial_clauses = []
        init_clause_lengths = np.zeros(shape=num_proofs, dtype=np.int32)
        proofs_chosen = []

        batch_size = num_proofs * (num_training_clauses + num_init_clauses)

        # Collect all clauses
        for p in range(num_proofs):
            proof_ind = self.get_proof()
            proof = self.proof_loader[proof_ind]
            proof.shuffle_conversion()
            proofs_chosen.append(proof_ind)
            # Prepare initial clauses
            ic = proof.get_init_clauses(num_init_clauses)
            initial_clauses.append(ic)
            init_clause_lengths[p] = min(proof.get_number_init_clauses(), num_init_clauses)
            # One negated conjecture per proof
            batch_neg_conj.append(proof.get_negated_conjecture())
            if proof.get_number_of_negatives() == 0:
                pos_next = np.zeros(shape=num_training_clauses) + 1
            elif proof.get_number_of_positives() == 0:
                pos_next = np.zeros(shape=num_training_clauses)
            else:
                # pos_next = np.random.choice(a=[0, 1], size=num_training_clauses, p=[1 - self.prob_pos, self.prob_pos])
                # No randomness, because it does not help the network and the loss function is optimized for exactly
                # self.prob_pos*num_training_clauses positive clauses
                pos_next = np.zeros(shape=num_training_clauses)
                pos_next[:int(math.ceil(self.prob_pos*num_training_clauses))] = 1
            for c in range(num_training_clauses):
                if pos_next[c] == 1:
                    training_clauses.append(proof.get_positive())
                    labels[p*num_training_clauses+c] = LABEL_POSITIVE
                else:
                    training_clauses.append(proof.get_negative())
                    labels[p*num_training_clauses+c] = LABEL_NEGATIVE
        # Augment all clauses
        initial_clauses = [self.augmenter.augment_clause(clause) for sublist in initial_clauses for clause in sublist]
        training_clauses = [self.augmenter.augment_clause(clause) for clause in training_clauses]
        batch_neg_conj = [self.augmenter.augment_clause(clause) for clause in batch_neg_conj]
        batch_clauses = training_clauses + initial_clauses

        batch_clause_length = get_clause_lengths(batch_clauses)
        batch_neg_conj_length = get_clause_lengths(batch_neg_conj)
        max_batch_clause_length = max(batch_clause_length)
        max_batch_neg_conj_length = max(batch_neg_conj_length)
        clause_batch = np.zeros(shape=[batch_size, max_batch_clause_length], dtype=np.int32) + self.empty_char
        neg_conj_batch = np.zeros(shape=[num_proofs, max_batch_neg_conj_length], dtype=np.int32) + self.empty_char
        batch_clause_length = np.array(batch_clause_length)
        batch_neg_conj_length = np.array(batch_neg_conj_length)
        for b in range(batch_size):
            clause_batch[b, :batch_clause_length[b]] = np.array(batch_clauses[b])
        for b in range(num_proofs):
            neg_conj_batch[b, :batch_neg_conj_length[b]] = np.array(batch_neg_conj[b])
        if max_batch_clause_length > self.max_clause_len:
            clause_batch = clause_batch[:, :self.max_clause_len]
            batch_clause_length = np.minimum(batch_clause_length, self.max_clause_len)
        if max_batch_neg_conj_length > self.max_neg_conj_len:
            neg_conj_batch = neg_conj_batch[:, :self.max_neg_conj_len]
            batch_neg_conj_length = np.minimum(batch_neg_conj_length, self.max_neg_conj_len)
        self.global_batch = [clause_batch, batch_clause_length, neg_conj_batch, batch_neg_conj_length,
                             init_clause_lengths, labels, proofs_chosen]

    def print_statistic(self):
        ClauseLoader.print_loader_statistic(self.proof_loader)

    def add_proof_index(self, pindex):
        if self.proof_indices.count(pindex) < 30:
            insert_index = randint(self.proof_index + 64 if self.proof_index + 64 < len(self.proof_loader) else 0,
                                   min(self.proof_index+384, len(self.proof_loader)))
            self.proof_indices.insert(insert_index, pindex)

    def remove_proof_index(self, pindex):
        if self.proof_indices.count(pindex) > 1:
            last_occ = len(self.proof_indices) - 1 - self.proof_indices[::-1].index(pindex)
            del self.proof_indices[last_occ]

    def set_max_len(self, clause_len, conj_len):
        self.max_clause_len = clause_len
        self.max_neg_conj_len = conj_len


if __name__ == "__main__":
    a = InitialClauseLoader(convert_to_absolute_path("/home/phillip/datasets/Cluster/Training/ClauseWeight_",
                                                     get_TPTP_train_small()))
    batch = a.get_batch(num_proofs=4, num_training_clauses=32, num_init_clauses=32)
    print("="*50+"\nClauses: "+str(batch[0].shape)+"\n"+str(batch[0]))
    print("="*50+"\nClauses length:"+str(batch[1].shape)+"\n"+str(batch[1]))
    print("="*50+"\nNegated conjecture: "+str(batch[2].shape)+"\n"+str(batch[2]))
    # print("="*50+"\nNegated conjecture length:"+str(batch[3].shape)+"\n"+str(batch[3]))
    # print("="*50+"\nInitial clause lengths: "+str(batch[4].shape)+"\n"+str(batch[4]))
    # print("="*50+"\nLabels:"+str(batch[5].shape)+"\n"+str(batch[5]))
