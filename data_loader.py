from collections import OrderedDict
from random import shuffle

import numpy as np


class ProofExampleLoader:
    def __init__(self, file_prefix):
        self.negated_conjecture = []
        self.pos_examples = []
        self.neg_examples = []
        self.neg_conjecture = []
        self.pos_indices = []
        self.neg_indices = []
        self.pos_index = 0
        self.neg_index = 0
        with open(file_prefix + "_pos.txt", "r") as f:
            for line in f:
                line = line.split(",")
                if line:
                    line = [int(i) for i in line]
                    self.pos_examples.append(line)
        with open(file_prefix + "_neg.txt", "r") as f:
            for line in f:
                line = line.split(",")
                if line:
                    line = [int(i) for i in line]
                    self.neg_examples.append(line)
        with open(file_prefix + "_conj.txt", "r") as f:
            for line in f:
                self.neg_conjecture = [int(i) for i in line.split(",")]

    def permute_positives(self):
        self.pos_indices = np.random.permutation(len(self.pos_examples))
        self.pos_index = 0

    def permute_negatives(self):
        self.neg_indices = np.random.permutation(len(self.neg_examples))
        self.neg_index = 0

    def get_positive(self):
        if self.pos_index >= len(self.pos_examples):
            self.permute_positives()
        c = self.pos_examples[self.pos_index]
        self.pos_index += 1
        return c

    def get_negative(self):
        if self.neg_index >= len(self.neg_examples):
            self.permute_negatives()
        c = self.neg_examples[self.neg_index]
        self.neg_index += 1
        return c

    def get_negated_conjecture(self):
        return self.neg_conjecture

    def get_number_of_positives(self):
        return len(self.pos_examples)

    def get_number_of_negatives(self):
        return len(self.neg_examples)

    def get_average_clause_length(self):
        pos_length = sum([len(c) for c in self.pos_examples]) / len(self.pos_examples)
        neg_length = sum([len(c) for c in self.neg_examples]) / len(self.neg_examples)
        return [pos_length, neg_length]

    def get_clause_statistic(self):
        pos_length = [len(c) for c in self.pos_examples]
        neg_length = [len(c) for c in self.neg_examples]
        pos_dict = OrderedDict(sorted((x, pos_length.count(x)) for x in set(pos_length)))
        neg_dict = OrderedDict(sorted((x, neg_length.count(x)) for x in set(neg_length)))
        return [pos_dict, neg_dict]


def get_clause_lengths(conj_list):
    return [len(c) for c in conj_list]


class ClauseLoader:
    def __init__(self, file_list, empty_char=0):
        self.empty_char = empty_char
        self.proof_loader = []
        self.proof_pos_indices = []
        self.proof_neg_indices = []
        self.pos_index = 0
        self.neg_index = 0
        self.pos_next = True
        for proof_file in file_list:
            self.proof_loader.append(ProofExampleLoader(proof_file))
        for index in range(len(self.proof_loader)):
            self.proof_pos_indices = self.proof_pos_indices + [index for _ in range(
                self.proof_loader[index].get_number_of_positives())]
            self.proof_neg_indices = self.proof_neg_indices + [index for _ in range(
                self.proof_loader[index].get_number_of_negatives())]
            print("Index " + str(index) + ": positives = " + str(
                self.proof_loader[index].get_number_of_positives()) + ", negatives = " + str(
                self.proof_loader[index].get_number_of_negatives()))
            print(self.proof_loader[index].get_average_clause_length())
            print(self.proof_loader[index].get_clause_statistic())

        self.permute_positives()
        self.permute_negatives()

    def permute_positives(self):
        shuffle(self.proof_pos_indices)
        self.pos_index = 0

    def permute_negatives(self):
        shuffle(self.proof_neg_indices)
        self.neg_index = 0

    def get_positive(self):
        if self.pos_index >= len(self.proof_pos_indices):
            self.permute_positives()
        c = self.proof_loader[self.proof_pos_indices[self.pos_index]]
        self.pos_index += 1
        return c

    def get_negative(self):
        if self.neg_index >= len(self.proof_neg_indices):
            self.permute_negatives()
        c = self.proof_loader[self.proof_neg_indices[self.neg_index]]
        self.neg_index += 1
        return c

    def get_batch(self, batch_size):
        batch_clauses = []
        batch_neg_conj = []
        for c in range(batch_size):
            if self.pos_next:
                proof = self.get_positive()
                batch_clauses.append(proof.get_positive())
                batch_neg_conj.append(proof.get_negated_conjecture())
            else:
                proof = self.get_negative()
                batch_clauses.append(proof.get_negative())
                batch_neg_conj.append(proof.get_negated_conjecture())
            self.pos_next = not self.pos_next
        batch_clause_length = get_clause_lengths(batch_clauses)
        batch_neg_conj_length = get_clause_lengths(batch_neg_conj)
        clause_batch = np.zeros(shape=[batch_size, max(batch_clause_length)]) + self.empty_char
        neg_conj_batch = np.zeros(shape=[batch_size, max(batch_neg_conj_length)]) + self.empty_char
        batch_clause_length = np.array(batch_clause_length)
        batch_neg_conj_length = np.array(batch_neg_conj_length)
        for b in range(batch_size):
            clause_batch[b, :batch_clause_length[b]] = np.array(batch_clauses[b])
            neg_conj_batch[b, :batch_neg_conj_length[b]] = np.array(batch_neg_conj[b])
        return [clause_batch, batch_clause_length, neg_conj_batch, batch_neg_conj_length]


a = ClauseLoader(["clause_data/example", "clause_data/example2"])
print(a.proof_pos_indices)
print(a.proof_neg_indices)
print(a.get_batch(128))
