# Easy proof generator like:  func(a) | func(X), func2(X) | func2(X), func3(X, b) | not func3(a,b)
# Length of proof variable. Should be smaller at the beginning, maximal function depth of 3
# Learn difference of func(X1, X2) and func(X1, X1)
# False ones by different constants and wrong path, like func(X), func10(a)...

import copy
import json
import random

from numpy.random import choice

from CNN_embedder_network import CNNEmbedder


class GeneratedVocab:

    def __init__(self, vocab_name, vocab_id, vocab_arity, is_negated=False):
        self.name = vocab_name
        self.id = vocab_id
        self.arity = vocab_arity
        self.depth = 0 if vocab_arity == 0 else -1
        self.is_negated = is_negated
        self.var_value = None
        self.parameters = list()

    def add_var_value(self, vocab_value):
        self.var_value = vocab_value

    def add_parameter(self, param):
        self.parameters.append(param)
        if self.arity < len(self.parameters):
            print("[!] WARNING: Got more parameters than expected. Vocab: " + self.name + ", Arity: " + str(
                self.arity) + ", Parameters: " + str(len(self.parameters)))
        self.depth = max(self.depth, param.depth)

    def change_to(self, new_vocab):
        self.name = new_vocab.name
        self.id = new_vocab.id
        self.arity = new_vocab.arity
        self.depth = new_vocab.depth
        self.is_negated = new_vocab.is_negated
        self.parameters = copy.deepcopy(new_vocab.parameters)

    def check_equality(self, vocab_to_check):
        if self.id == vocab_to_check.id:
            params_equal = True
            for param_index in range(len(self.parameters)):
                params_equal = params_equal and self.parameters[param_index].check_equality(
                    vocab_to_check.parameters[param_index])
            return params_equal
        else:
            return False

    def replace_subvocab(self, vocab_to_replace, new_vocab):
        if self.check_equality(vocab_to_replace):
            self.change_to(new_vocab)
        else:
            for param in self.parameters:
                param.replace_subvocab(vocab_to_replace, new_vocab)

    def replace_variables_by_value(self):
        if self.var_value is not None:
            self.change_to(self.var_value)
        else:
            for params in self.parameters:
                params.replace_variables_by_value()

    def get_all_variables(self):
        if self.arity == -1:
            return [self]
        else:
            var_list = list()
            for params in self.parameters:
                var_list = var_list + params.get_all_variables()
            return var_list

    def vocab_to_id_list(self, vocab_dict_by_name):
        func_list = list()
        if self.is_negated:
            func_list.append(vocab_dict_by_name["~"])
        func_list.append(self.id)
        if self.arity > 0:
            func_list.append(vocab_dict_by_name["("])
            for param in self.parameters:
                func_list = func_list + param.vocab_to_id_list(vocab_dict_by_name)
            func_list.append(vocab_dict_by_name[")"])
        return func_list

    def vocab_to_str(self):
        s = ""
        if self.is_negated:
            s += "~"
        s += self.name
        if self.arity > 0:
            s += "("
            for param in self.parameters:
                if s[-1] != "(":
                    s += ","
                s += param.vocab_to_str()
            s += ")"
        return s


class GeneratedClause:

    def __init__(self, vocabs):
        self.vocabs = vocabs

    def append_vocab(self, voc):
        self.vocabs.append(voc)

    def get_all_variables(self):
        var_list = list()
        for voc in self.vocabs:
            var_list = var_list + voc.get_all_variables()
        unique_vars = list()
        for var in var_list:
            contains_var = False
            for unique_var in unique_vars:
                if unique_var.id == var.id:
                    contains_var = True
                    break
            if not contains_var:
                unique_vars.append(var)
        return unique_vars

    def clause_to_id_list(self, vocab_dict_by_name):
        clause_list = list()
        for voc in self.vocabs:
            if len(clause_list) > 0:
                clause_list.append(vocab_dict_by_name[","])
            clause_list = clause_list + voc.vocab_to_id_list(vocab_dict_by_name)
        return clause_list

    def clause_to_str(self):
        s = ""
        for voc in self.vocabs:
            if len(s) > 0:
                s += " | "
            s += voc.vocab_to_str()
        return s


class ProofGenerator:

    def __init__(self, vocab_file="Conversion_vocab.txt"):
        self.vocab_dict_by_arity = None
        self.vocab_dict_by_name = None
        self.vocab_dict_by_id = None
        self.arity_indices = None
        self.arity_distribution = None
        self.max_arity = -1
        self.free_arities = None

        self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        with open(vocab_file) as f:
            self.vocab_dict_by_name = json.load(f)
        self.vocab_dict_by_id = {v: k for k, v in self.vocab_dict_by_name.items()}
        self.vocab_dict_by_arity = dict()
        for voc_name, voc_id in self.vocab_dict_by_name.items():
            arity = CNNEmbedder.get_arity_from_vocab(voc_name)
            if arity >= -1:
                if arity not in self.vocab_dict_by_arity:
                    self.vocab_dict_by_arity[arity] = list()
                self.vocab_dict_by_arity[arity].append(voc_id)
            self.max_arity = max(self.max_arity, arity)
        self.arity_distribution = [2 ** (-k) for k in range(self.max_arity + 1)]

    def generate_new_proof(self, proof_length, amount_negatives=24):
        self.arity_indices = dict()
        self.free_arities = dict()
        for k, v in self.vocab_dict_by_arity.items():
            random.shuffle(v)
            self.arity_indices[k] = 0
            self.free_arities[k] = len(v) > 0
        used_vocab = list()
        neg_conj = self.generate_neg_conj()
        positive_clauses = self.generate_clause_queue(neg_conj, proof_length)
        # negative_clauses = self.generate_negative_clauses(used_vocab, amount_negatives)
        # init_clauses = positive_clauses + negative_clauses
        return positive_clauses

    def generate_neg_conj(self):
        neg_conj = self.get_random_function(2, 4)
        neg_conj.is_negated = True
        return neg_conj

    def generate_clause_queue(self, neg_conj, proof_len):
        # Think about: Work with variables and constants
        # Think about: Work with equals instead of ors
        all_positive_clauses = [GeneratedClause([neg_conj])]
        last_voc = copy.deepcopy(neg_conj)
        for proof_ind in range(proof_len - 1):
            self.arity_indices[-1] = 0
            last_voc.is_negated = False
            all_positive_clauses.append(GeneratedClause([last_voc]))
            next_voc = self.get_random_function(2, 4)
            next_voc.is_negated = True
            new_clause = GeneratedClause([last_voc, next_voc])
            all_positive_clauses.append(new_clause)
            last_voc = copy.deepcopy(next_voc)
        last_voc.is_negated = False
        all_positive_clauses.append(GeneratedClause([last_voc]))
        return all_positive_clauses

    def get_next_free_vocab(self, arity):
        if self.free_arities[arity]:
            vocab = self.vocab_dict_by_arity[arity][self.arity_indices[arity]]
            self.arity_indices[arity] = self.arity_indices[arity] + 1
            self.free_arities[arity] = (len(self.vocab_dict_by_arity[arity]) - self.arity_indices[arity]) > 0
        else:
            vocab = None
        return vocab

    def get_random_free_vocab(self, arity):
        if self.free_arities[arity]:
            voc_index = random.randint(self.arity_indices[arity], len(self.vocab_dict_by_arity[arity]))
            vocab = self.vocab_dict_by_arity[arity][voc_index]
        else:
            vocab = None
        return vocab

    def create_vocab(self, voc_id, arity):
        return GeneratedVocab(vocab_name=self.vocab_dict_by_id[voc_id], vocab_id=voc_id, vocab_arity=arity)

    def create_vocab_by_name(self, voc_name):
        voc_id = self.vocab_dict_by_name(voc_name)
        voc_arity = CNNEmbedder.get_arity_from_vocab(voc_name)
        return GeneratedVocab(vocab_name=voc_name, vocab_id=voc_id, vocab_arity=voc_arity)

    def get_random_function(self, min_depth=0, max_depth=4):
        print("Random function between (" + str(min_depth) + "," + str(max_depth) + ")")
        reached_final_depth = (min_depth <= 0)
        arity = self.get_random_arity(min_arity=min(min_depth, 1), max_arity=4 if max_depth > 0 else 0)
        voc_id = self.get_next_free_vocab(arity)
        vocab = self.create_vocab(voc_id, arity)
        if arity > 0:
            for param in range(arity):
                new_min_depth = max(min_depth - 1 if not reached_final_depth else 0, 0)
                new_param = self.get_random_function(min_depth=new_min_depth, max_depth=max_depth - 1)
                vocab.add_parameter(new_param)
                reached_final_depth = reached_final_depth or vocab.depth >= min_depth
        return vocab

    def get_random_arity(self, min_arity=0, max_arity=-1, var_probs=0.5):
        if max_arity < 0:
            max_arity = self.max_arity

        dist = self.arity_distribution[min_arity:max_arity + 1]
        possible_arities = range(min_arity, max_arity + 1)
        print("Possible arity: " + str(possible_arities) + ", distribution: " + str(dist))
        random_arity = choice(possible_arities, p=[p / sum(dist) for p in dist])
        print("My arity: " + str(random_arity) + " between (" + str(min_arity) + "," + str(max_arity) + ")")
        if random_arity == 0 and (choice([0, 1], p=[1 - var_probs, var_probs]) == 1):
            random_arity = -1

        if not self.free_arities[random_arity]:
            random_arity = None
            possible_arities = range(min_arity if min_arity > 0 else -1, max_arity)
            random.shuffle(possible_arities)
            for pos_arity in possible_arities:
                if not self.free_arities[pos_arity]:
                    random_arity = pos_arity
                    break
            if random_arity is None:
                print("[!] ERROR: No free vocab available...")
        return random_arity


if __name__ == "__main__":
    proof_gen = ProofGenerator("Conversion_vocab.txt")
    pos_clauses = proof_gen.generate_new_proof(5)
    for clause_index, clause in enumerate(pos_clauses):
        print("Clause ["+str(clause_index)+"]: "+clause.clause_to_str())
    # random_vocab = proof_gen.get_random_function(2, 3)
    # print("Random function: " + str(random_vocab.vocab_to_str()))
