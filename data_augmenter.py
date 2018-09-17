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

    def get_additional_arguments(self, clause):
        return None

    def create_vocab_augmentation_dict(self, proof_vocab):
        return None

    def augment_vocab(self, clauses, augm_dict):
        return clauses


class DataAugmenter:

    def __init__(self, use_conversion=False):
        self.use_conversion = use_conversion
        self.vocab = self.load_vocab()
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.arity_vocab = {k: CNNEmbedder.get_arity_from_vocab(v) for k, v in self.reversed_vocab.items()}
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

    def create_vocab_augmentation_dict(self, proof_vocab):
        vocab_augmentation = np.random.choice(a=[0, 1, 2], p=[0.9, 0.1, 0.0]) # 0.3, 0.4, 0.3
        conv_vocab = dict()
        for voc in proof_vocab:
            new_voc = voc
            voc_arity = self.arity_vocab[voc]
            if vocab_augmentation != 0 and voc_arity >= 0:
                augment_voc = np.random.choice(a=[True, False], p=[0.2, 0.8]) or (vocab_augmentation == 2)
                if augment_voc:
                    new_voc = self._augment_random_vocab(voc)
            conv_vocab[voc] = new_voc
        return conv_vocab

    def augment_vocab(self, clauses, augm_dict):
        for i in range(len(clauses)):
            if isinstance(clauses[i], list):
                clauses[i] = [augm_dict[c] if (c in augm_dict) else c for c in clauses[i]]
            else:
                clauses[i] = augm_dict[clauses[i]] if clauses[i] in augm_dict else clauses[i]
        return clauses

    def _augment_random_vocab(self, voc):
        voc_arity = self.arity_vocab[voc]
        rand_index = np.random.randint(len(self.vocab_by_arity[voc_arity]))
        new_voc = self.vocab[self.vocab_by_arity[voc_arity][rand_index]]
        return new_voc

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
        arities = [self.arity_vocab[c]+2 for c in clause]
        clause_layers = self.get_clause_layers(clause)
        literal_ids, literal_length_activation, positions, pos_neg_literal = self.get_literal_inputs(clause)
        clause_length_activation = math.tanh(3 * (len(clause)-30) / (len(clause)+30))
        clause_len_input = [clause_length_activation for _ in range(len(clause))]
        np_arities = DataAugmenter.convert_to_one_hot(arities, max_val=7, periodic=False)
        np_clause_layers = DataAugmenter.convert_to_one_hot(clause_layers, max_val=5, periodic=False)
        np_clause_len = np.array(clause_len_input, dtype=np.float32)
        np_literal_ids = DataAugmenter.convert_to_one_hot(literal_ids, max_val=3, periodic=True)
        np_pos_neg_literal = np.array(pos_neg_literal, dtype=np.float32)
        np_literal_length = np.array(literal_length_activation, dtype=np.float32)
        np_positions = np.array(positions, dtype=np.float32)
        add_input = np.stack([np_clause_len, np_pos_neg_literal, np_literal_length, np_positions], axis=1)
        add_input = np.concatenate([np_arities, np_clause_layers, np_literal_ids, add_input], axis=1)
        return add_input

    @staticmethod
    def convert_to_one_hot(array, max_val=None, periodic=False):
        if max_val is None:
            max_val = max(array)
        one_hot_vector = np.zeros(shape=[len(array), max_val + 1], dtype=np.float32)
        for i in range(len(array)):
            if array[i] < 0:
                continue
            index = array[i]
            if periodic:
                index = index % max_val
            else:
                index = min(index, max_val)
            one_hot_vector[i][index] = 1
        return one_hot_vector

    def get_literal_inputs(self, clause):
        ids = list()
        lengths = list()
        positions = list()
        neg_literals = [-1]
        current_id = 0
        last_literal_change = -1
        for i in range(len(clause)):
            if clause[i] == self.vocab[","]:
                current_id += 1
                lengths.append((i - 1) - last_literal_change)
                if lengths[-1] == 0:
                    print("[!] ERROR: Zero length found at "+str(i)+" -> clause "+str(clause))
                elif lengths[-1] == 1:
                    positions[last_literal_change+1] = 0
                else:
                    positions[last_literal_change+1:] = [2*(p*1.0/(lengths[-1]-1)) - 1 for p in positions[last_literal_change+1:]]
                last_literal_change = i
                ids.append(-1)
                positions.append(0)
                neg_literals.append(-1)
            else:
                ids.append(current_id)
                positions.append((i - 1) - last_literal_change)
                if clause[i] == self.vocab["~"]:
                    neg_literals[-1] = 1
        lengths.append((len(clause) - 1) - last_literal_change)
        if lengths[-1] == 0:
            print("[!] ERROR: Zero length found at " + str(i) + " -> clause " + str(clause))
        elif lengths[-1] == 1:
            positions[last_literal_change+1] = 0
        else:
            positions[last_literal_change+1:] = [2*(p*1.0/(lengths[-1]-1)) - 1 for p in positions[last_literal_change+1:]]

        number_of_literals = current_id + 1
        clause_length = len(clause) - (number_of_literals - 1)  # Abziehen der ","
        length_activations = [math.tanh(3.0 * (l * 1.0 / clause_length - 1.0 / number_of_literals) / (l * 1.0 / clause_length + 1.0 / number_of_literals)) for l in lengths]
        length_input = list()
        pos_neg_literal = list()
        for i in range(number_of_literals):
            length_input += [length_activations[i] for _ in range(lengths[i])]
            pos_neg_literal += [neg_literals[i] for _ in range(lengths[i])]
            length_input.append(0)
            pos_neg_literal.append(0)
        del length_input[-1]
        del pos_neg_literal[-1]

        return ids, length_input, positions, pos_neg_literal

    def get_clause_layers(self, clause):
        layers = list()
        current_layer = 0
        for i in range(len(clause)):
            if clause[i] == self.vocab[")"]:
                current_layer -= 1
            layers.append(current_layer)
            if clause[i] == self.vocab["("]:
                current_layer += 1
        if current_layer != 0:
            print("[!] WARNING: Clause did not end on layer 0 but "+str(current_layer) + " -> clause: " + str(clause))
        return layers

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
        for i in range(25):
            var = "X" + str(i + 1)
            var_list.append(vocab[var])
        return var_list


def test_augmentation():
    vocab = CNNEmbedder.get_vocabulary(use_conversion=False)
    rev_vocab = {v: k for k, v in vocab.items()}
    clause = ["~", "left_inverse#1", "(", "X1", "X2", ")", ",", "left_zero#2", "(", "lonely#1", "(", "X1", ")", "X3", ")", "=", "living#2", "(", "X4", "X1", ")",",","lonely#1","(","X4",")"]
    clause = [vocab[x] for x in clause]
    proof_vocab = list(np.unique(np.array(clause)))
    print(clause)
    aug = DataAugmenter()
    # print(aug.get_clause_layers(clause))
    # print(aug.get_literal_inputs(clause))
    # print(aug.get_additional_arguments(clause))
    augm_dict = aug.create_vocab_augmentation_dict(proof_vocab)
    print(augm_dict)
    print(aug.augment_vocab(clause, augm_dict))
    # print(aug.vocab_by_arity)
    # for i in range(10):
    #     print("Arity = "+str(i)+": "+str(len(aug.vocab_by_arity[i])))
    # print(aug.augment_clause(clause))
    # start_time = time.time()
    # for _ in range(16):
    #     aug_clause = aug.augment_positive_to_negative([], np.unique(np.array(clause)))
    # print("Time: " + str((time.time() - start_time) / 16))
    # print([rev_vocab[x] for x in aug_clause])


if __name__ == '__main__':
    test_augmentation()
