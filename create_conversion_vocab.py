import json
import numpy as np
from random import shuffle
import pprint


def create_conversion_vocab(arity_dict, file_name="Conversion_vocab.txt"):
    vocab = dict()
    vocab_id = 0
    vocab_id = add_special_characters(vocab, vocab_id)
    for arity, vocab_amount in arity_dict.items():
        if arity >= -1:
            default_name = ""
            if arity == 0:
                default_name = "const_"
            elif arity > 0:
                default_name = "func_"+str(arity)+"_"
            elif arity == -1:
                default_name = "X"
            for vocab_index in range(vocab_amount):
                vocab_name = default_name + str(vocab_index+1)
                if arity >= 0:
                    vocab_name += "#" + str(arity)
                vocab[vocab_name] = vocab_id
                vocab_id += 1
    with open(file_name, 'w') as f:
        json.dump(vocab, f, indent=4, sort_keys=True)


def add_special_characters(vocab_dict, start_id):
    special_characters = ["(", ")", ",", "=", "<-", " ", "~"]
    for c in special_characters:
        vocab_dict[c] = start_id
        start_id += 1
    return start_id


def generate_vocab_variables(vocab_size, embedding_size, min_diffs=32, min_commons=0):
    all_vocabs = list()
    start_vocab = [1 for _ in range(int(embedding_size * 3 / 8))] + \
                  [0 for _ in range(int(embedding_size * 2 / 8))] + \
                  [-1 for _ in range(int(embedding_size * 3 / 8))]
    all_vocabs.append(start_vocab)
    for i in range(vocab_size - 1):
        found_new_vocab = False
        voc_array = np.array(all_vocabs)
        while not found_new_vocab:
            new_voc = start_vocab[:]
            shuffle(new_voc)
            new_voc_array = np.array(new_voc)
            diffs = np.sum(np.not_equal(voc_array, new_voc_array), axis=1)

            if np.all(diffs >= min_diffs) and np.all((embedding_size - diffs) >= min_commons):
                all_vocabs.append(new_voc)
                found_new_vocab = True
    return all_vocabs


def test_vocab_generation():
    vocs = generate_vocab_variables(300, 256, min_diffs=128, min_commons=64)
    print("\nVocabulary\n"+"="*100)
    for v in vocs:
        print(v)


if __name__ == "__main__":
    # create_conversion_vocab({-1: 25, 0: 75, 1: 75, 2: 75, 3: 10, 4: 10, 5: 10, 6: 5, 7: 2, 8: 2})
    test_vocab_generation()
