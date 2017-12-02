from glob import glob
import os
from enum import Enum
from CNN_embedder_network import CNNEmbedder

# FIFO - /home/phillip/datasets/Cluster/*/*/E-Prover_TF_Very_Silent___E---2.0_G----_0001_C18_F1_SE_CS_SP_S00/*/*.txt
# ClauseWeight - /home/phillip/datasets/Cluster/*/*/E-Prover_TF_Very_Silent___E---2.0_G----_0003_C18_F1_SE_CS_SP_S0Y/*/*.txt
# Best - /home/phillip/datasets/Cluster/*/*/E-Prover_TF_Very_Silent___E---2.0_G-E--_208_C18_F1_SE_CS_SP_PS_S0Y/*/*.txt

ALL_FILES = sorted(glob("/home/phillip/datasets/Cluster/Job25255_output/*/E-Prover_TF_Very_Silent___E---2.0_G----_0003_C18_F1_SE_CS_SP_S0Y/*/*.txt"))
OUTPUT_DIR = "/home/phillip/datasets/Cluster/Parsed/"
PREFIX = "ClauseWeight_"
VOCAB_CODES = CNNEmbedder.get_vocabulary().values()
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class FileStatus(Enum):
    PRE_PROCESSING = 0
    POSITIVE_EXAMPLES = 1
    NEGATIVE_EXAMPLES = 2
    POST_PROCESSING = 3
    ERROR = 4


def test_line(line_text):
    chars = line_text.split(",")
    if chars[0] == '':
        return False
    if 'Got FunCode out of bounds' in chars[0]:
        print("[!] ERROR: "+line_text)
        return True
    clause = [int(i) for i in chars]
    for fun_code in clause:
        if fun_code not in VOCAB_CODES:
            print("[!] ERROR: Found code that is not in vocabulary: "+str(fun_code)+" (see "+line_text+")")
            return True
    return False


wrong_vocab = 0
resource_out = 0
for file in ALL_FILES:
    print("Extracting "+file+"...")
    file_name = PREFIX + file.split("/")[-2].split(".")[0]
    output_pos_file = os.path.join(OUTPUT_DIR, file_name+"_pos.txt")
    output_neg_file = os.path.join(OUTPUT_DIR, file_name+"_neg.txt")
    with open(file, 'r') as f:
        f_neg = open(output_neg_file, 'w')
        f_pos = open(output_pos_file, 'w')
        status = FileStatus.PRE_PROCESSING
        for line in f:
            text = line.split('\t')[-1].split('\n')[0]
            # print("Status "+str(status)+", Line "+text)
            if status == FileStatus.POST_PROCESSING:
                break
            if status == FileStatus.POSITIVE_EXAMPLES:
                if "# Training: Positive examples end" not in text:
                    f_pos.write(text+"\n")
                    if test_line(text):
                        wrong_vocab += 1
                        status = FileStatus.ERROR
                        break
                else:
                    status = FileStatus.PRE_PROCESSING

            elif status == FileStatus.NEGATIVE_EXAMPLES:
                if "# Training: Negative examples end" not in text:
                    f_neg.write(text+"\n")
                    test_line(text)
                    if test_line(text):
                        wrong_vocab += 1
                        status = FileStatus.ERROR
                        break
                else:
                    status = FileStatus.POST_PROCESSING

            elif status == FileStatus.PRE_PROCESSING:
                if "# Training: Positive examples begin" in text:
                    status = FileStatus.POSITIVE_EXAMPLES
                elif "# Training: Negative examples begin" in text:
                    status = FileStatus.NEGATIVE_EXAMPLES
                elif "EOF" in text:
                    print("Found "+text+" while pre-processing")
                    resource_out += 1
                    status = FileStatus.ERROR
                    break
        f_neg.close()
        f_pos.close()
        if status == FileStatus.ERROR:
            os.remove(output_pos_file)
            os.remove(output_neg_file)
print("Found "+str(wrong_vocab)+" files with vocab problems")
print("Found "+str(resource_out)+" files with resource out")