import json


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


if __name__ == "__main__":
    create_conversion_vocab({-1: 25, 0: 75, 1: 75, 2: 75, 3: 10, 4: 10, 5: 10, 6: 5, 7: 2, 8: 2})
