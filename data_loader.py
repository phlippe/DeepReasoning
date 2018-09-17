from collections import OrderedDict
from random import shuffle
from glob import glob

import sys
import numpy as np
import math
import json
import os
from CNN_embedder_network import CNNEmbedder

from data_augmenter import DataAugmenter, DefaultAugmenter
from multiprocessing import Pool
from TPTP_train_val_files import *
if sys.version_info[0] < 3:
    print(" [!] ERROR: Could not import module thread. Python version "+str(sys.version_info))
    sys.exit(1)
    #from thread import start_new_thread
else:
    from _thread import start_new_thread
    #pass

LABEL_POSITIVE = 0
LABEL_NEGATIVE = 1
IS_UNPROCESSED_INDEX = "is_unp"
EXAMPLE_INDEX = "index"
BATCH_NEGATIVE_INDEX = "neg"
BATCH_POSITIVE_INDEX = "pos"
USE_CONVERSION = False


class ProofExampleLoader:
    def __init__(self, file_prefix=None, use_conversion=False):
        self.pos_examples = list()
        self.neg_examples = list()
        self.init_clauses = list()
        self.neg_conjecture = list()
        self.pos_indices = list()
        self.neg_indices = list()
        self.unp_indices = list()
        self.current_batches = list()
        self.pos_index = 0
        self.neg_index = 0
        self.unp_index = 0
        self.prefix = file_prefix
        self.all_vocabs = None
        self.base_vocab = None
        self.vocabs_by_arity = None
        self.vocab_conversion_dict = None
        self.active_conversion = None
        self.use_conversion = use_conversion

        if file_prefix is not None:
            self.pos_examples = ProofExampleLoader.read_file(file_prefix, "pos")
            self.neg_examples = ProofExampleLoader.read_file(file_prefix, "neg")
            self.unp_examples = ProofExampleLoader.read_file(file_prefix, "unp")
            self.init_clauses = ProofExampleLoader.read_file(file_prefix, "init")

            self.pos_loss_list = np.array([-1.0 for _ in self.pos_examples])
            self.neg_loss_list = np.array([-1.0 for _ in self.neg_examples] + [-1.0 for _ in self.unp_examples])

            self.p_neg = (self.get_number_of_negatives() + self.get_number_of_unprocessed() / 4.0) * 1.0 / (self.get_number_of_negatives() + self.get_number_of_unprocessed() + 1e-10)
            self.permute_positives()
            self.permute_negatives()
            self.permute_unprocessed()

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
                                print("[!] ERROR CONJ: "+str(e))
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
                print("[!] ERROR CONJ: "+str(e))
                self.neg_conjecture = []

            self.create_own_vocab()
            if self.use_conversion:
                try:
                    self.analyse_vocab()
                    self.shuffle_conversion()
                except IndexError as e:
                    print("[!] ERROR CONVERSION: "+str(e))

    @staticmethod
    def read_file(file_prefix, file_postfix):
        all_lines = list()
        file_to_read = file_prefix + "_"+file_postfix+".txt"
        if os.path.isfile(file_to_read):
            with open(file_to_read, "r") as f:
                for line in f:
                    line = line.split(",")
                    if line and line[0] is not '\n' and len(line) <= 150:
                        try:
                            line = [int(i) for i in line]
                            all_lines.append(line)
                        except ValueError as e:
                            print("[!] ERROR "+file_postfix.upper()+": "+str(e))
        return all_lines

    def new_batch(self):
        new_batch = dict()
        new_batch[BATCH_POSITIVE_INDEX] = list()
        new_batch[BATCH_NEGATIVE_INDEX] = list()
        self.current_batches.append(new_batch)

    def add_example_to_batch(self, index, example, is_unprocessed=False):
        if index == BATCH_NEGATIVE_INDEX:
            self.current_batches[-1][index].append(
                {EXAMPLE_INDEX: example, IS_UNPROCESSED_INDEX: is_unprocessed})
        elif index == BATCH_POSITIVE_INDEX:
            self.current_batches[-1][index].append(example)
        else:
            print("[!] ERROR: Could not add clause example to batch. Unknown index: "+str(index))

    def loss_to_batch(self, loss_list):
        batch_to_process = self.current_batches[0]
        number_of_positives = len(batch_to_process[BATCH_POSITIVE_INDEX])
        # print("Number of positives: "+str(number_of_positives))
        # print("Number of negatives overall: "+str(self.get_number_of_negatives()))
        # print("Loss list: "+str(loss_list))
        for i in range(number_of_positives):
            pos_index = batch_to_process[BATCH_POSITIVE_INDEX][i]
            self.pos_loss_list[pos_index] = loss_list[i]
        for i in range(len(loss_list) - number_of_positives):
            neg_index = batch_to_process[BATCH_NEGATIVE_INDEX][i][EXAMPLE_INDEX]
            # print("Neg index: "+str(neg_index))
            is_unprocessed = batch_to_process[BATCH_NEGATIVE_INDEX][i][IS_UNPROCESSED_INDEX]
            if is_unprocessed:
                self.neg_loss_list[self.get_number_of_negatives()+neg_index] = loss_list[i+number_of_positives]
            else:
                self.neg_loss_list[neg_index] = loss_list[i+number_of_positives]
        del self.current_batches[0]

    def get_argmax_negatives(self, number_of_clauses):
        argmax_list = np.argsort(self.neg_loss_list)[-number_of_clauses:]
        clause_list = list()
        for ind in argmax_list:
            if ind >= self.get_number_of_negatives():
                self.add_example_to_batch(BATCH_NEGATIVE_INDEX, ind-self.get_number_of_negatives(), is_unprocessed=True)
                clause_list.append(self.unp_examples[ind-self.get_number_of_negatives()])
            else:
                self.add_example_to_batch(BATCH_NEGATIVE_INDEX, ind, is_unprocessed=False)
                clause_list.append(self.neg_examples[ind])
        return clause_list

    def get_argmax_positives(self, number_of_clauses):
        argmax_list = np.argsort(self.pos_loss_list)[-number_of_clauses:]
        clause_list = list()
        for ind in argmax_list:
            self.add_example_to_batch(BATCH_POSITIVE_INDEX, ind)
            clause_list.append(self.pos_examples[ind])
        return clause_list

    def permute_positives(self):
        self.pos_indices = np.random.permutation(len(self.pos_examples))
        self.pos_index = 0

    def permute_negatives(self):
        self.neg_indices = np.random.permutation(len(self.neg_examples))
        self.neg_index = 0

    def permute_unprocessed(self):
        self.unp_indices = np.random.permutation(len(self.unp_examples))
        self.unp_index = 0

    def get_positive(self):
        if self.pos_index >= len(self.pos_examples):
            self.permute_positives()
        c = self.pos_examples[self.pos_indices[self.pos_index]]
        self.add_example_to_batch(BATCH_POSITIVE_INDEX, self.pos_indices[self.pos_index])
        if self.use_conversion:
            c = [self.active_conversion[voc] for voc in c]
        self.pos_index += 1
        return c

    def get_negative(self):
        rand_choice = np.random.choice([0, 1], size=1, p=[self.p_neg, 1 - self.p_neg])
        if (rand_choice == 0 or self.get_number_of_unprocessed() == 0) and self.get_number_of_negatives() > 0:
            if self.neg_index >= len(self.neg_examples):
                self.permute_negatives()
            c = self.neg_examples[self.neg_indices[self.neg_index]]
            self.add_example_to_batch(BATCH_NEGATIVE_INDEX, self.neg_indices[self.neg_index], is_unprocessed=False)
            self.neg_index += 1
        else:
            if self.unp_index >= len(self.unp_examples):
                self.permute_unprocessed()
            c = self.unp_examples[self.unp_indices[self.unp_index]]
            self.add_example_to_batch(BATCH_NEGATIVE_INDEX, self.unp_indices[self.unp_index], is_unprocessed=True)
            self.unp_index += 1
        if self.use_conversion:
            c = [self.active_conversion[voc] for voc in c]
        return c

    def get_init_clauses(self, size):
        shuffle(self.init_clauses)
        if len(self.init_clauses) >= size:
            all_inits = self.init_clauses[:size]
        else:
            all_inits = self.init_clauses + [[5, 5, 5, 5, 5] for _ in range(size - len(self.init_clauses))]
        if self.use_conversion:
            all_inits = [[self.active_conversion[voc] for voc in clause] for clause in all_inits]
        return all_inits

    def get_number_init_clauses(self):
        return len(self.init_clauses)

    def get_negated_conjecture(self):
        neg_conj = self.neg_conjecture
        if self.use_conversion:
            neg_conj = [self.active_conversion[voc] for voc in neg_conj]
        return neg_conj

    def get_number_of_positives(self):
        return len(self.pos_examples)

    def get_number_of_negatives(self):
        return len(self.neg_examples)

    def get_number_of_unprocessed(self):
        return len(self.unp_examples)

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

    def create_own_vocab(self):
        all_pos_vocabs = np.unique(np.array([c for clause in self.pos_examples for c in clause]))
        all_neg_vocabs = np.unique(np.array([c for clause in self.neg_examples for c in clause]))
        all_init_vocabs = np.unique(np.array([c for clause in self.init_clauses for c in clause]))
        all_neg_conj_vocabs = np.unique(np.array(self.neg_conjecture))
        self.all_vocabs = list(
            np.unique(np.concatenate([all_pos_vocabs, all_neg_vocabs, all_init_vocabs, all_neg_conj_vocabs], axis=0)))

    def get_vocab(self):
        return self.all_vocabs

    def analyse_vocab(self):
        with open("Vocabs.txt", 'r') as vocab_file:
            base_vocab = json.load(vocab_file)
        self.base_vocab = {v: k for k, v in base_vocab.items()}

        self.vocabs_by_arity = dict()
        for voc in self.all_vocabs:
            arity = CNNEmbedder.get_arity_from_vocab(self.base_vocab[voc])
            if arity >= -1:
                if arity not in self.vocabs_by_arity:
                    self.vocabs_by_arity[arity] = list()
                self.vocabs_by_arity[arity].append(voc)

        with open("Conversion_vocab.txt") as f:
            conv_dict = json.load(f)
        self.vocab_conversion_dict = dict()
        for voc_name, voc_id in conv_dict.items():
            arity = CNNEmbedder.get_arity_from_vocab(voc_name)
            if arity >= -1:
                if arity not in self.vocab_conversion_dict:
                    self.vocab_conversion_dict[arity] = list()
                self.vocab_conversion_dict[arity].append(voc_id)

        self.active_conversion = dict()
        self.active_conversion[base_vocab[" "]] = conv_dict[" "]
        for voc in self.all_vocabs:
            arity = CNNEmbedder.get_arity_from_vocab(self.base_vocab[voc])
            if arity < -1:
                voc_name = self.base_vocab[voc]
                conv_id = conv_dict[voc_name]
                self.active_conversion[voc] = conv_id

    def shuffle_conversion(self):
        if self.use_conversion:
            for arity, voc_list in self.vocab_conversion_dict.items():
                if arity in self.vocabs_by_arity:
                    shuffle(voc_list)
                    for voc_index in range(len(self.vocabs_by_arity[arity])):
                        self.active_conversion[self.vocabs_by_arity[arity][voc_index]] = voc_list[voc_index]

    def get_arity_distribution(self):
        arity_dict = dict()
        for voc in self.all_vocabs:
            arity = CNNEmbedder.get_arity_from_vocab(self.base_vocab[voc])
            if arity in arity_dict:
                arity_dict[arity] += 1
            else:
                arity_dict[arity] = 1
        # print("My arity dict: "+str(arity_dict))
        return arity_dict


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
    def add_single_file_to_proof_loader(proof_file):
        print(proof_file)
        new_proof_loader = ProofExampleLoader(proof_file, use_conversion=USE_CONVERSION)
        if len(
                new_proof_loader.get_negated_conjecture()) == 0 or new_proof_loader.get_number_of_negatives() == 0 or new_proof_loader.get_number_of_positives() == 0:
            print("Could not use this proof loader, no negatives and positives or no conjecture...")
            # problems[0] = problems[0] + 1
            return None
        elif len(new_proof_loader.get_negated_conjecture()) > 150:
            print("Too large negated conjecture. Will not be used...")
            # problems[1] = problems[1] + 1
            return None
        elif new_proof_loader.get_number_init_clauses() == 0:
            print("No init clauses provided...")
            # problems[2] = problems[2] + 1
            return None
        elif new_proof_loader.get_number_init_clauses() > 150:
            print("Too many init clauses...")
            # problems[3] = problems[3] + 1
            return None
        elif new_proof_loader.get_number_of_positives() > 150:
            print("Too many positive clauses...")
            # problems[4] = problems[4] + 1
            return None
        # elif new_proof_loader.get_number_of_negatives() < 10:
        #     print("Less than 10 negative clauses. Might be too easy...")
        #     easy_problems = easy_problems + 1
        #     print("Easy problems: "+str(easy_problems))
        else:
            print("Add proof loader to data loader")
            # proof_loader.append(new_proof_loader)
            return new_proof_loader
            # if new_proof_loader.get_number_init_clauses() > 32:
            #    large_inits += 1

    @staticmethod
    def initialize_proof_loader(file_list, use_conversion=False):
        global USE_CONVERSION
        USE_CONVERSION = use_conversion
        problems = [0, 0, 0, 0, 0]
        large_inits = 0
        easy_problems = 0
        proof_loader = []
        # for proof_file in file_list:

        with Pool() as p:
            proof_loader = p.map(ClauseLoader.add_single_file_to_proof_loader, file_list)
        proof_loader = [loader for loader in proof_loader if loader is not None]

        print("="*50)
        print("Overall proof loader: "+str(len(proof_loader))+" | "+str(len(file_list)))
        print("Found following problems:")
        print("No neg/pos/neg_conj: "+str(problems[0]))
        print("Too large neg_conj: "+str(problems[1]))
        print("No init clauses: "+str(problems[2]))
        print("Too many init clauses: "+str(problems[3]))
        print("Too many positive clauses: "+str(problems[4]))
        print("Too less negatives: "+str(easy_problems))
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
        overall_unp = 0
        for loader in proof_loader:
            overall_neg += loader.get_number_of_negatives()
            overall_pos += loader.get_number_of_positives()
            overall_unp += loader.get_number_of_unprocessed()
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
        print("Unprocessed: "+str(overall_unp))
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


def test_proof_loader():
    proof_loader = ProofExampleLoader("clause_data/example2")
    for _ in range(2):
        proof_loader.new_batch()
        print("Loss list: "+str(proof_loader.loss_list))
        clauses = list()
        for _ in range(8):
            clauses.append(proof_loader.get_positive())
        for _ in range(24):
            clauses.append(proof_loader.get_negative())
        for i in range(len(clauses)):
            print("Clause "+str(i)+": "+str(clauses[i]))
    proof_loader.loss_to_batch(np.array([a*1.0/32.0 for a in range(32)]))
    proof_loader.loss_to_batch(np.array([a*1.0/32.0 for a in range(32)]))
    print("Loss list: "+str(proof_loader.loss_list))
    proof_loader.new_batch()
    clauses = list()
    for _ in range(8):
        clauses.append(proof_loader.get_positive())
    argmax_clauses = proof_loader.get_argmax_negatives(8)
    for i in range(len(argmax_clauses)):
        print("Argmax clause "+str(i)+": "+str(argmax_clauses[i]))




if __name__ == "__main__":
    test_proof_loader()
    # train_loader = ClauseLoader(convert_to_absolute_path("/home/phillip/datasets/Cluster/Training/ClauseWeight_", get_TPTP_train_files(Dataset.Best)))
    # val_loader = ClauseLoader(convert_to_absolute_path("/home/phillip/datasets/Cluster/Training/ClauseWeight_", get_TPTP_test_files(Dataset.Best)))
    # all_proof_loader = train_loader.proof_loader + val_loader.proof_loader
    # overall_vocab_use = dict()
    # for loader in all_proof_loader:
    #     ar_dict = loader.get_arity_distribution()
    #     for k, v in ar_dict.items():
    #         if k in overall_vocab_use:
    #             overall_vocab_use[k] = max(overall_vocab_use[k], v)
    #         else:
    #             overall_vocab_use[k] = v
    #         if v > 100:
    #             print("Found greater 100 ("+loader.prefix+")")
    # print("Number of loaders: "+str(len(all_proof_loader)))
    # print("Overall vocab dict: "+str(overall_vocab_use))
