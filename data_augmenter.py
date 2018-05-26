import itertools
import numpy as np
import time
import math

from CNN_embedder_network import CNNEmbedder

MAX_ARITY = 10


class DefaultAugmenter:
    def __init__(self):
        pass

    def augment_clause(self, clause):
        return clause

    def augment_positive_to_negative(self, clause, proof_vocab):
        return clause


class DataAugmenter:

    def __init__(self, use_conversion=False):
        self.use_conversion = use_conversion
        self.vocab = self.load_vocab()
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_by_arity = self.sort_vocab(self.vocab)
        self.arity_distribution = [len(self.vocab_by_arity[i]) for i in range(len(self.vocab_by_arity)) if i > 0]
        self.arity_distribution = [x * 1.0 / sum(self.arity_distribution) for x in self.arity_distribution]
        self.variables = DataAugmenter.get_variables(self.vocab)

    def augment_clause(self, clause):
        aug_clause = list(clause)
        self.permute_literals(aug_clause, ",")
        self.reverse_equations(aug_clause, "=", ",")
        aug_clause = self.change_variables(aug_clause)
        return aug_clause

    def permute_literals(self, clause, vocab_char):
        split_clause = DataAugmenter.split_list(clause, (self.vocab[vocab_char],))
        if len(split_clause) > 1:
            indices = np.random.permutation(len(split_clause))
            start_index = 0
            for i in range(len(split_clause)):
                lit = split_clause[indices[i]]
                clause[start_index:start_index + len(lit)] = lit
                start_index += len(lit) + 1
                if i < len(split_clause) - 1:
                    clause[start_index - 1] = self.vocab[vocab_char]

    def reverse_equations(self, clause, eq_vocab_char, lit_vocab_char):
        split_clause = DataAugmenter.split_list(clause, (self.vocab[lit_vocab_char],))
        start_index = 0
        for i in range(len(split_clause)):
            equation_split = DataAugmenter.split_list(split_clause[i], (self.vocab[eq_vocab_char],))
            if len(equation_split) > 1:
                indices = np.random.permutation(len(equation_split))
                eq_start_index = 0
                for j in range(len(equation_split)):
                    eq_parts = equation_split[indices[j]]
                    clause[start_index+eq_start_index:start_index+eq_start_index + len(eq_parts)] = eq_parts
                    eq_start_index += len(eq_parts) + 1
                    if j < len(equation_split) - 1:
                        clause[start_index + eq_start_index - 1] = self.vocab[eq_vocab_char]
            start_index += len(split_clause[i]) + 1

    def change_variables(self, clause):
        max_vocab = 0
        for i in range(len(self.variables)):
            var = "X" + str(i + 1)
            if self.vocab[var] in clause:
                max_vocab += 1
            else:
                break
        var_permutation = self.random_variables()
        for i in range(max_vocab):
            clause = [var_permutation[i]-200 if x == self.variables[i] else x for x in clause]
        for i in range(max_vocab):
            clause = [var_permutation[i] if x == var_permutation[i]-200 else x for x in clause]
        return clause

    def random_variables(self):
        permute_indices = np.random.permutation(len(self.variables))
        return [self.variables[i] for i in permute_indices]

    def augment_positive_to_negative(self, clause, proof_vocab):
        clause = self.add_random_literals(clause, proof_vocab)
        clause = self.augment_clause(clause)
        return clause

    def add_random_literals(self, clause, proof_vocab):
        number_of_literals = np.random.randint(2, 10, 1)
        if len(clause) > 0:
            random_literal_prob = np.random.choice(a=[0.6, 0.0], p=[0.75, 0.25])
        else:
            random_literal_prob = 1.0
        string_vocab = {self.reversed_vocab[x]: x for x in proof_vocab}
        arity_vocab = self.sort_vocab(string_vocab)
        arity_dist = [len(arity_vocab[i]) for i in range(len(arity_vocab)) if i > 0]
        arity_dist = [x * 1.0 / sum(arity_dist) for x in arity_dist]
        for _ in range(int(number_of_literals)):
            use_rand_lit = np.random.choice(a=[True, False], p=[random_literal_prob, 1 - random_literal_prob])
            if use_rand_lit:
                lit = self.get_random_different_literal(proof_vocab, 0)
            else:
                lit = self.get_random_same_literal(proof_vocab, arity_vocab, arity_dist)
            if len(clause) > 0:
                clause = clause + [self.vocab[","]]
            clause = clause + lit
        return clause

    def get_random_different_literal(self, proof_vocab, current_depth):
        lit = list()
        if current_depth == 0:
            arity = np.random.choice(a=np.arange(1, MAX_ARITY), p=self.arity_distribution)
            counter = 0
            lit_id = proof_vocab[0]
            while counter == 0 or (counter < 100 and lit_id in proof_vocab):
                lit_name = self.vocab_by_arity[arity][np.random.randint(0, len(self.vocab_by_arity[arity]))]
                lit_id = self.vocab[lit_name]
                counter = counter + 1
        else:
            arity_dist = np.power(2.0, - 1.0 * np.arange(0, MAX_ARITY) * current_depth)
            arity_dist = [x / sum(arity_dist) for x in arity_dist]
            arity = np.random.choice(a=np.arange(0, MAX_ARITY), p=arity_dist)

            use_var = np.random.choice(a=[True, False], p=[0.5, 0.5])
            if arity > 0 or not use_var:
                lit_name = self.vocab_by_arity[arity][np.random.randint(0, len(self.vocab_by_arity[arity]))]
            else:
                lit_name = "X"+str(np.random.randint(1, 30))
            lit_id = self.vocab[lit_name]   # String to ID

        lit.append(lit_id)
        if arity > 0:
            lit.append(self.vocab["("])
            for i in range(arity):
                param = self.get_random_different_literal(proof_vocab, current_depth+1)
                lit = lit + param
            lit.append(self.vocab[")"])
        return lit

    def get_random_same_literal(self, proof_vocab, proof_arity_vocab, arity_distribution):
        lit = list()
        arity = np.random.choice(a=np.arange(1, MAX_ARITY), p=arity_distribution)
        lit_name = proof_arity_vocab[arity][np.random.randint(0, len(proof_arity_vocab[arity]))]
        lit_id = self.vocab[lit_name]

        lit.append(lit_id)
        lit.append(self.vocab["("])
        for i in range(arity):
            param = self.get_random_different_literal(proof_vocab, 1)
            lit = lit + param
        lit.append(self.vocab[")"])
        return lit

    def get_additional_arguments(self, clause):
        #self.get_clause_layers()
        number_of_literals = 5
        literal_length = 2
        literal_length_activation = math.tanh(3 * (literal_length - number_of_literals / clause) / (literal_length + number_of_literals / clause))
        clause_length_activation = math.tanh(3 * (len(clause)-30) / (len(clause)+30))
        return list()


    @staticmethod
    def split_list(iterable, splitters):
        return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]

    def load_vocab(self):
        return CNNEmbedder.get_vocabulary(use_conversion=self.use_conversion)

    def sort_vocab(self, vocab):
        vocab_by_arity = dict()
        orig_vocab = [k.split('#') for k, v in vocab.items()]
        orig_vocab = [x for x in orig_vocab if len(x) > 1]
        for i in range(MAX_ARITY):
            vocab_by_arity[i] = [x[0] + "#" + x[1] for x in orig_vocab if x[1] == str(i)]
        return vocab_by_arity

    @staticmethod
    def get_variables(vocab):
        var_list = []
        for i in range(30):
            var = "X" + str(i + 1)
            var_list.append(vocab[var])
        return var_list


def test_augmentation():
    vocab = CNNEmbedder.get_vocabulary(use_conversion=False)
    rev_vocab = {v: k for k, v in vocab.items()}
    clause = ["left_inverse#1", "(", "X1", "X2", ")", ",", "left_zero#2", "(", "X2", "X3", ")", "=", "living#2", "(", "X4", "X1", ")",",","lonely#1","(","X4",")"]
    clause = [vocab[x] for x in clause]
    print(clause)
    aug = DataAugmenter()
    print(aug.vocab_by_arity)
    for i in range(10):
        print("Arity = "+str(i)+": "+str(len(aug.vocab_by_arity[i])))
    print(aug.augment_clause(clause))
    start_time = time.time()
    for _ in range(16):
        aug_clause = aug.augment_positive_to_negative([], np.unique(np.array(clause)))
    print("Time: " + str((time.time() - start_time) / 16))
    print([rev_vocab[x] for x in aug_clause])


if __name__ == '__main__':
    test_augmentation()
