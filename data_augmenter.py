import itertools
import numpy as np
import time

from CNN_embedder_network import CNNEmbedder


class DefaultAugmenter:
    def augment_clause(self, clause):
        return clause


class DataAugmenter:
    def __init__(self):
        self.vocab = DataAugmenter.load_vocab()
        self.variables = DataAugmenter.get_variables(self.vocab)

    def augment_clause(self, clause):
        aug_clause = list(clause)
        self.permute_literals(aug_clause)
        aug_clause = self.change_variables(aug_clause)
        return aug_clause

    def permute_literals(self, clause):
        split_clause = DataAugmenter.split_list(clause, (self.vocab[","],))
        indices = np.random.permutation(len(split_clause))
        start_index = 0
        for i in range(len(split_clause)):
            lit = split_clause[indices[i]]
            clause[start_index:start_index + len(lit)] = lit
            start_index += len(lit) + 1
            if i < len(split_clause) - 1:
                clause[start_index - 1] = self.vocab[","]

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
            clause = [var_permutation[i] if x == self.variables[i] else x for x in clause]
        return clause

    def random_variables(self):
        permute_indices = np.random.permutation(len(self.variables))
        return [self.variables[i] for i in permute_indices]

    @staticmethod
    def split_list(iterable, splitters):
        return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]

    @staticmethod
    def load_vocab():
        return CNNEmbedder.get_vocabulary()

    @staticmethod
    def get_variables(vocab):
        var_list = []
        for i in range(50):
            var = "X" + str(i + 1)
            var_list.append(vocab[var])
        return var_list


def test_augmentation():
    vocab = CNNEmbedder.get_vocabulary()
    clause = ["rotate", "(", "X1", "X2", ")", ",", "flip", "(", "X2", "X3", ")", ",", "flip", "(", "X4", "X1", ")"]
    clause = [vocab[x] for x in clause]
    print(clause)
    aug = DataAugmenter()
    start_time = time.time()
    for i in range(8192):
        aug_clause = aug.augment_clause(clause)
    print("Time: " + str(time.time() - start_time))


if __name__ == '__main__':
    test_augmentation()
