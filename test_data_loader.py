from data_loader import ClauseLoader, get_clause_lengths

import math
import numpy as np
from TPTP_train_val_files import *
from data_augmenter import DataAugmenter


LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1


class TestClauseLoader:

    def __init__(self, file_list, empty_char=5, max_clause_len=150, max_neg_conj_len=-1, use_conversion=False):
        self.empty_char = int(empty_char)
        self.proof_loader = []
        self.max_clause_len = max_clause_len
        if max_neg_conj_len <= 0:
            self.max_neg_conj_len = self.max_clause_len
        else:
            self.max_neg_conj_len = max_neg_conj_len

        self.proof_loader = ClauseLoader.initialize_proof_loader(
            [f for f in file_list if 'ClauseWeight_LCL' not in f], use_conversion=use_conversion)
        self.augmenter = DataAugmenter(use_conversion=use_conversion)

    def get_all_batches(self, num_proofs, num_training_clauses, num_init_clauses):
        global LABEL_NEGATIVE, LABEL_POSITIVE
        all_batches = []
        sub_batch_index = 0
        labels = np.zeros(shape=num_proofs*num_training_clauses, dtype=np.int32)
        batch_neg_conj = []
        training_clauses = []
        initial_clauses = []
        init_clause_lengths = np.zeros(shape=num_proofs, dtype=np.int32)
        proofs_chosen = []
        batch_size = num_proofs * (num_training_clauses + num_init_clauses)

        for p in range(len(self.proof_loader)):
            proof = self.proof_loader[p]
            # Prepare initial clauses
            ic = proof.get_init_clauses(num_init_clauses)

            proof_clauses = []
            clause_labels = []
            for _ in range(proof.get_number_of_positives()):
                proof_clauses.append(proof.get_positive())
                clause_labels.append(LABEL_POSITIVE)
            for _ in range(proof.get_number_of_negatives()):
                proof_clauses.append(proof.get_negative())
                clause_labels.append(LABEL_NEGATIVE)
            for _ in range(proof.get_number_of_positives()):
                proof_clauses.append(self.augmenter.augment_positive_to_negative(proof.get_positive(), proof.get_vocab()))
                clause_labels.append(LABEL_NEGATIVE)
            for _ in range(10):
                proof_clauses.append(self.augmenter.augment_positive_to_negative([], proof.get_vocab()))
                clause_labels.append(LABEL_NEGATIVE)
            needed_sub_batches = int(math.ceil(len(proof_clauses) / num_training_clauses))
            for s in range(needed_sub_batches):
                if sub_batch_index >= num_proofs:
                    all_batches.append([initial_clauses, training_clauses, batch_neg_conj, init_clause_lengths,
                                        proofs_chosen, labels])
                    initial_clauses = []
                    labels = np.zeros(shape=num_proofs*num_training_clauses, dtype=np.int32)
                    training_clauses = []
                    batch_neg_conj = []
                    init_clause_lengths = np.zeros(shape=num_proofs, dtype=np.int32)
                    proofs_chosen = []
                    sub_batch_index = 0

                proofs_chosen.append(p)
                initial_clauses.append(ic)
                init_clause_lengths[sub_batch_index] = min(proof.get_number_init_clauses(), num_init_clauses)
                batch_neg_conj.append(proof.get_negated_conjecture())

                for c in range(num_training_clauses):
                    label_index = sub_batch_index * num_training_clauses + c
                    if len(proof_clauses) > 0:
                        training_clauses.append(proof_clauses[0])
                        labels[label_index] = clause_labels[0]
                        del proof_clauses[0]
                        del clause_labels[0]
                    else:
                        training_clauses.append([self.empty_char, self.empty_char, self.empty_char])
                        labels[label_index] = -1  # ATTENTION: DO NOT PUT THESE LABELS IN LOSS FUNCTION!

                sub_batch_index += 1

        for s in range(sub_batch_index, num_proofs+1):
            if s >= num_proofs:
                all_batches.append([initial_clauses, training_clauses, batch_neg_conj, init_clause_lengths,
                                    proofs_chosen, labels])
                break
            proofs_chosen.append(-1)
            initial_clauses.append(initial_clauses[-1])
            init_clause_lengths[s] = init_clause_lengths[s-1]
            batch_neg_conj.append(batch_neg_conj[-1])
            for c in range(num_training_clauses):
                training_clauses.append(training_clauses[-1])
                labels[s * num_training_clauses + c] = -1

        processed_batches = []
        for batch in all_batches:
            initial_clauses, training_clauses, batch_neg_conj, init_clause_lengths, proofs_chosen, labels = batch
            # Augment all clauses
            initial_clauses = [clause for sublist in initial_clauses for clause in sublist]
            batch_clauses = training_clauses + initial_clauses

            batch_clause_length = get_clause_lengths(batch_clauses)
            batch_neg_conj_length = get_clause_lengths(batch_neg_conj)
            clause_batch = np.zeros(shape=[batch_size, max(batch_clause_length)], dtype=np.int32) + self.empty_char
            neg_conj_batch = np.zeros(shape=[num_proofs, max(batch_neg_conj_length)],
                                      dtype=np.int32) + self.empty_char
            batch_clause_length = np.array(batch_clause_length)
            batch_neg_conj_length = np.array(batch_neg_conj_length)
            for b in range(batch_size):
                clause_batch[b, :batch_clause_length[b]] = np.array(batch_clauses[b])
            for b in range(num_proofs):
                neg_conj_batch[b, :batch_neg_conj_length[b]] = np.array(batch_neg_conj[b])
            if max(batch_clause_length) > self.max_clause_len:
                clause_batch = clause_batch[:, :self.max_clause_len]
                batch_clause_length = np.minimum(batch_clause_length, self.max_clause_len)
            if max(batch_neg_conj_length) > self.max_neg_conj_len:
                neg_conj_batch = neg_conj_batch[:, :self.max_neg_conj_len]
                batch_neg_conj_length = np.minimum(batch_neg_conj_length, self.max_neg_conj_len)
            processed_batches.append([clause_batch, batch_clause_length, neg_conj_batch, batch_neg_conj_length,
                                      init_clause_lengths, labels, proofs_chosen])
        return processed_batches

    def print_out_results(self, batches, results):
        s = ""
        for index in range(len(batches)):
            clause_batch, _, _, _, _, labels, proofs_chosen = batches[index]
            training_clauses = clause_batch[:int(clause_batch.shape[0] / 2)]
            for proof in range(len(proofs_chosen)):
                proof_index = proofs_chosen[proof]
                if proof_index != -1:
                    proof_prefix = self.proof_loader[proof_index].prefix.split('_', -1)[-1]
                    clause_per_proof = int(training_clauses.shape[0] / len(proofs_chosen))
                    for c in range(clause_per_proof):
                        clause_index = proof * clause_per_proof + c
                        if labels[clause_index] != -1:
                            s += proof_prefix + " -> Weight: "+"%4.3f" % (results[index][clause_index])+" / " +\
                                 str(labels[clause_index])+" | "+self.clause_to_string(training_clauses[clause_index])+\
                                 "\n"
        return s

    def clause_to_string(self, clause):
        s = "["
        for i in range(len(clause)):
            if clause[i] != self.empty_char:
                if i > 0:
                    s += ","
                s += str(clause[i])
        s += "]"
        return s


if __name__ == '__main__':
    a = TestClauseLoader(convert_to_absolute_path("/home/phillip/datasets/Cluster/Training/ClauseWeight_",
                                                  get_TPTP_clause_test_files(Dataset.Best)))
    b = a.get_all_batches(4, 32, 32)
    print("First labels:"+str(b[0][5]))
    print(a.print_out_results(b, [[1 for _ in range(128)] for _ in range(len(b))]))