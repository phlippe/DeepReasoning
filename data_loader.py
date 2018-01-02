from collections import OrderedDict
from random import shuffle
from glob import glob

import sys
import numpy as np
import math

from data_augmenter import DataAugmenter, DefaultAugmenter
from TPTP_train_val_files import *
if sys.version_info[0] < 3:
    pass
    # from thread import start_new_thread
else:
    from _thread import start_new_thread

LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1


class ProofExampleLoader:
    def __init__(self, file_prefix):
        self.pos_examples = []
        self.neg_examples = []
        self.init_clauses = []
        self.neg_conjecture = []
        self.pos_indices = []
        self.neg_indices = []
        self.pos_index = 0
        self.neg_index = 0
        self.prefix = file_prefix

        self.pos_examples = ProofExampleLoader.read_file(file_prefix, "pos")
        self.neg_examples = ProofExampleLoader.read_file(file_prefix, "neg")
        self.init_clauses = ProofExampleLoader.read_file(file_prefix, "init")

        try:
            with open(file_prefix + "_conj.txt", "r") as f:
                c = 0
                last_line = None
                for line in f:
                    full_line = line
                    line = line.split("\n")[0].split(",")
                    if line and line[0] is not '\n' and not (full_line == last_line) and "6" in line:
                        last_conj = self.neg_conjecture[:]
                        try:
                            if c > 0:
                                self.neg_conjecture.append(2)  # Appending ,

                            self.neg_conjecture = self.neg_conjecture + [int(i) for i in line]
                            c += 1
                        except ValueError as e:
                            print("[!] ERROR CONJ: "+e.message)
                            self.neg_conjecture = last_conj
                    elif "6" not in line:
                        print("[!] WARNING: Conjecture line without \"not\" removed: "+str(line))
                    last_line = full_line
                if len(self.neg_conjecture) == 0 and last_line is not None:
                    line = last_line.split("\n")[0].split(",")
                    if not (len(line) == 1 and line[0] == ''):
                        print("[!] WARNING: Using last line of conjecture: "+str(line))
                        self.neg_conjecture = self.neg_conjecture + [int(i) for i in line]
        except IOError as e:
            print("[!] ERROR CONJ: "+e.message)
            self.neg_conjecture = []

    @staticmethod
    def read_file(file_prefix, file_postfix):
        all_lines = []
        with open(file_prefix + "_"+file_postfix+".txt", "r") as f:
            for line in f:
                line = line.split(",")
                if line and line[0] is not '\n':
                    try:
                        line = [int(i) for i in line]
                        all_lines.append(line)
                    except ValueError as e:
                        print("[!] ERROR "+file_postfix.upper()+": "+e.message)
        return all_lines

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

    def get_init_clauses(self, size):
        shuffle(self.init_clauses)
        if len(self.init_clauses) >= size:
            return self.init_clauses[:size]
        else:
            return self.init_clauses + [[5, 5, 5, 5, 5] for _ in range(size - len(self.init_clauses))]

    def get_number_init_clauses(self):
        return len(self.init_clauses)

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
    def __init__(self, file_list, empty_char=5, augment=True, prob_pos=0.3):
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

        self.proof_loader = ClauseLoader.initialize_proof_loader(file_list=file_list)

        for index in range(len(self.proof_loader)):
            self.proof_pos_indices = self.proof_pos_indices + [index for _ in range(
                int(math.ceil(math.sqrt(self.proof_loader[index].get_number_of_positives()))**1.5))]

            self.proof_neg_indices = self.proof_neg_indices + [index for _ in range(
                int(math.ceil(math.sqrt(self.proof_loader[index].get_number_of_negatives()))**1.5))]
            print("Index " + str(index) + ": positives = " + str(
                self.proof_loader[index].get_number_of_positives()) + ", negatives = " + str(
                self.proof_loader[index].get_number_of_negatives()) + "( "+self.proof_loader[index].prefix+" )")
            # print(self.proof_loader[index].get_average_clause_length())
            # print(self.proof_loader[index].get_clause_statistic())

        self.permute_positives()
        self.permute_negatives()

    @staticmethod
    def initialize_proof_loader(file_list):
        problems = [0, 0, 0]
        large_inits = 0
        proof_loader = []
        for proof_file in file_list:
            print(proof_file)
            new_proof_loader = ProofExampleLoader(proof_file)
            if len(new_proof_loader.get_negated_conjecture()) == 0 or (new_proof_loader.get_number_of_negatives() == 0 and new_proof_loader.get_number_of_positives() == 0):
                print("Could not use this proof loader, no negatives and positives or no conjecture...")
                problems[0] += 1
            elif len(new_proof_loader.get_negated_conjecture()) > 150:
                print("Too large negated conjecture. Will not be used...")
                problems[1] += 1
            elif new_proof_loader.get_number_init_clauses() == 0:
                print("No init clauses provided...")
                problems[2] += 1
            else:
                proof_loader.append(new_proof_loader)
                if new_proof_loader.get_number_init_clauses() > 32:
                    large_inits += 1

        print("="*50)
        print("Found following problems:")
        print("No neg/pos/neg_conj: "+str(problems[0]))
        print("Too large neg_conj: "+str(problems[1]))
        print("No init clauses: "+str(problems[2]))
        print("More init clauses than 32: "+str(large_inits))
        print("="*50)

        return proof_loader

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
        start_new_thread(self.__get_batch, (batch_size,))
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
        ClauseLoader.print_loader_statistic(self.proof_loader)

    @staticmethod
    def print_loader_statistic(proof_loader):
        overall_neg = 0
        overall_pos = 0
        for loader in proof_loader:
            overall_neg += loader.get_number_of_negatives()
            overall_pos += loader.get_number_of_positives()
        avg_pos_len = 0
        avg_neg_len = 0
        greatest_pos_len = 0
        greatest_neg_len = 0
        no_neg_conj = 0
        for loader in proof_loader:
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


if __name__ == "__main__":
    a = ClauseLoader(convert_to_absolute_path("/home/phillip/datasets/Cluster/Training/ClauseWeight_", get_TPTP_train_files()))
    for loader in a.proof_loader:
        print("Init clauses: "+str(loader.get_number_init_clauses()))
