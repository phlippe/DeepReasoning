import itertools
import numpy as np
import time

from CNN_embedder_network import CNNEmbedder


class DefaultAugmenter:
    def __init__(self):
        pass

    def augment_clause(self, clause):
        return clause


class DataAugmenter:
    def __init__(self, use_conversion=False):
        self.use_conversion = use_conversion
        self.vocab = self.load_vocab()
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

    @staticmethod
    def split_list(iterable, splitters):
        return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]

    def load_vocab(self):
        return CNNEmbedder.get_vocabulary(use_conversion=self.use_conversion)

    @staticmethod
    def get_variables(vocab):
        var_list = []
        for i in range(20):
            var = "X" + str(i + 1)
            var_list.append(vocab[var])
        return var_list


def test_augmentation():
    vocab = CNNEmbedder.get_vocabulary(use_conversion=False)
    clause = ["left_inverse#1", "(", "X1", "X2", ")", ",", "left_zero#2", "(", "X2", "X3", ")", "=", "living#2", "(", "X4", "X1", ")",",","lonely#1","(","X4",")"]
    clause = [vocab[x] for x in clause]
    print(clause)
    aug = DataAugmenter()
    print(aug.augment_clause(clause))
    start_time = time.time()
    #for i in range(8192):
    #    aug_clause = aug.augment_clause(clause)
    print("Time: " + str(time.time() - start_time))


if __name__ == '__main__':
    test_augmentation()
