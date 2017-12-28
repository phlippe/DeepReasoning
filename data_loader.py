from collections import OrderedDict
from random import shuffle
from glob import glob
import os

import numpy as np
import thread
import math

from data_augmenter import DataAugmenter, DefaultAugmenter

LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1


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
        self.prefix = file_prefix
        with open(file_prefix + "_pos.txt", "r") as f:
            for line in f:
                line = line.split(",")
                if line and line[0] is not '\n':
                    try:
                        line = [int(i) for i in line]
                        self.pos_examples.append(line)
                    except ValueError as e:
                        print("[!]ERROR POS: "+e.message)
        with open(file_prefix + "_neg.txt", "r") as f:
            for line in f:
                line = line.split(",")
                if line and line[0] is not '\n':
                    try:
                        line = [int(i) for i in line]
                        self.neg_examples.append(line)
                    except ValueError as e:
                        print("[!]ERROR NEG: "+e.message)
        try:
            with open(file_prefix + "_conj.txt", "r") as f:
                c = 0
                last_line = ""
                for line in f:
                    full_line = line
                    line = line.split("\n")[0].split(",")
                    if line and line[0] is not '\n' and not (full_line == last_line):
                        last_conj = self.neg_conjecture
                        try:
                            if c > 0:
                                self.neg_conjecture.append(2)  # Appending ,

                            self.neg_conjecture = self.neg_conjecture + [int(i) for i in line]
                            c += 1
                        except ValueError as e:
                            print("[!]ERROR CONJ: "+e.message)
                            self.neg_conjecture = last_conj
                    last_line = full_line
        except IOError as e:
            print("[!]ERROR CONJ: "+e.message)
            self.neg_conjecture = [5, 5, 5]

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
        pos_length = sum([len(c) for c in self.pos_examples]) / (len(self.pos_examples) if len(self.pos_examples) > 0 else 1)
        neg_length = sum([len(c) for c in self.neg_examples]) / (len(self.neg_examples) if len(self.neg_examples) > 0 else 1)
        return [pos_length, neg_length]

    def get_greatest_clause_length(self):
        pos_length = max([len(c) for c in self.pos_examples]) if len(self.pos_examples) > 0 else 0
        neg_length = max([len(c) for c in self.neg_examples]) if len(self.neg_examples) > 0 else 0
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
    def __init__(self, file_list, empty_char=0, augment=True, prob_pos=0.3):
        if augment:
            self.augmenter = DataAugmenter()
        else:
            self.augmenter = DefaultAugmenter()
        self.empty_char = empty_char
        self.prob_pos = prob_pos
        self.proof_loader = []
        self.proof_pos_indices = []
        self.proof_neg_indices = []
        self.pos_index = 0
        self.neg_index = 0
        self.pos_next = True
        self.global_batch = None
        for proof_file in file_list:
            print(proof_file)
            new_proof_loader = ProofExampleLoader(proof_file)
            if len(new_proof_loader.get_negated_conjecture()) == 0 or (new_proof_loader.get_number_of_negatives() == 0 and new_proof_loader.get_number_of_positives() == 0):
                print("Could not use this proof loader...")
            else:
                self.proof_loader.append(new_proof_loader)
        for index in range(len(self.proof_loader)):
            self.proof_pos_indices = self.proof_pos_indices + [index for _ in range(
                int(math.ceil(math.sqrt(self.proof_loader[index].get_number_of_positives()))))]

            self.proof_neg_indices = self.proof_neg_indices + [index for _ in range(
                int(math.ceil(math.sqrt(self.proof_loader[index].get_number_of_negatives()))))]
            print("Index " + str(index) + ": positives = " + str(
                self.proof_loader[index].get_number_of_positives()) + ", negatives = " + str(
                self.proof_loader[index].get_number_of_negatives()) + "( "+self.proof_loader[index].prefix+" )")
            # print(self.proof_loader[index].get_average_clause_length())
            # print(self.proof_loader[index].get_clause_statistic())

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
        if self.global_batch is None:
            self.__get_batch(batch_size=batch_size)
        current_batch = self.global_batch
        self.global_batch = None
        thread.start_new_thread(self.__get_batch, (batch_size,))
        return current_batch

    def __get_batch(self, batch_size):
        global LABEL_NEGATIVE, LABEL_POSITIVE
        batch_clauses = []
        batch_neg_conj = []
        labels = np.zeros(shape=batch_size)
        pos_next = np.random.choice(a=[0, 1], size=batch_size, p=[1 - self.prob_pos, self.prob_pos])
        for c in range(batch_size):
            if pos_next[c] == 1:
                proof = self.get_positive()
                batch_clauses.append(self.augmenter.augment_clause(proof.get_positive()))
                batch_neg_conj.append(self.augmenter.augment_clause(proof.get_negated_conjecture()))
                labels[c] = LABEL_POSITIVE
            else:
                proof = self.get_negative()
                batch_clauses.append(self.augmenter.augment_clause(proof.get_negative()))
                batch_neg_conj.append(self.augmenter.augment_clause(proof.get_negated_conjecture()))
                labels[c] = LABEL_NEGATIVE
        batch_clause_length = get_clause_lengths(batch_clauses)
        batch_neg_conj_length = get_clause_lengths(batch_neg_conj)
        clause_batch = np.zeros(shape=[batch_size, max(batch_clause_length)]) + self.empty_char
        neg_conj_batch = np.zeros(shape=[batch_size, max(batch_neg_conj_length)]) + self.empty_char
        batch_clause_length = np.array(batch_clause_length)
        batch_neg_conj_length = np.array(batch_neg_conj_length)
        for b in range(batch_size):
            clause_batch[b, :batch_clause_length[b]] = np.array(batch_clauses[b])
            neg_conj_batch[b, :batch_neg_conj_length[b]] = np.array(batch_neg_conj[b])
        if max(batch_clause_length) > 150:
            clause_batch = clause_batch[:, :150]
            batch_clause_length = np.minimum(batch_clause_length, 150)
        if max(batch_neg_conj_length) > 150:
            neg_conj_batch = neg_conj_batch[:, :150]
            batch_neg_conj_length = np.minimum(batch_neg_conj_length, 150)
        self.global_batch = [clause_batch, batch_clause_length, neg_conj_batch, batch_neg_conj_length, labels]

    def print_statistic(self):
        overall_neg = len(self.proof_neg_indices)
        overall_pos = len(self.proof_pos_indices)
        avg_pos_len = 0
        avg_neg_len = 0
        greatest_pos_len = 0
        greatest_neg_len = 0
        no_neg_conj = 0
        for loader in self.proof_loader:
            curr_avg = loader.get_average_clause_length()
            avg_pos_len += 1.0 * curr_avg[0] * loader.get_number_of_positives() / overall_pos
            avg_neg_len += 1.0 * curr_avg[1] * loader.get_number_of_negatives() / overall_neg
            curr_greatest = loader.get_greatest_clause_length()
            greatest_pos_len = max(greatest_pos_len, curr_greatest[0])
            greatest_neg_len = max(greatest_neg_len, curr_greatest[1])
            no_neg_conj += 1 if len(loader.get_negated_conjecture()) == 0 else 0
        print("="*50+"\nSTATISTICS")
        print("Negative: "+str(overall_neg))
        print("Average length: "+str(avg_neg_len))
        print("Greatest: "+str(greatest_neg_len))
        print("Positive: "+str(overall_pos))
        print("Average length: "+str(avg_pos_len))
        print("Greatest: "+str(greatest_pos_len))
        print("Without Negated Conjecture: "+str(no_neg_conj))

    @staticmethod
    def create_file_list_from_dir(directory):
        all_files = sorted(glob(directory + "*_neg.txt"))
        print("Found "+str(len(all_files))+" files in "+directory)
        return [f.rsplit('_', 1)[0] for f in all_files]


# a = ClauseLoader(ClauseLoader.create_file_list_from_dir("/home/phillip/datasets/Cluster/Training/ClauseWeight"))
# a.print_statistic()
# print(a.proof_pos_indices)
# print(a.proof_neg_indices)
# print(a.get_batch(128))
