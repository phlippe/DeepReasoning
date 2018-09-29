from glob import glob
import os
from enum import Enum
from CNN_embedder_network import CNNEmbedder
import _thread
import time
from multiprocessing import Pool


# FIFO - /home/phillip/datasets/Cluster/*/*/E-Prover_TF_Very_Silent___E---2.0_G----_0001_C18_F1_SE_CS_SP_S00/*/*.txt
# ClauseWeight - /home/phillip/datasets/Cluster/*/*/E-Prover_TF_Very_Silent___E---2.0_G----_0003_C18_F1_SE_CS_SP_S0Y/*/*.txt
# Best - /home/phillip/datasets/Cluster/*/*/E-Prover_TF_Very_Silent___E---2.0_G-E--_208_C18_F1_SE_CS_SP_PS_S0Y/*/*.txt

ALL_FILES = sorted(glob("/home/phillip/datasets/Cluster/Job25255_output/*/E-Prover_TF_Very_Silent___E---2.0_G----_0001_C18*/*/*.txt"))
OUTPUT_DIR = "/home/phillip/datasets/Cluster/Parsed/"
PREFIX = "FIFO_"
VOCAB_CODES = CNNEmbedder.get_vocabulary(use_conversion=False).values()
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class FileStatus(Enum):
    PRE_PROCESSING = 0
    POSITIVE_EXAMPLES = 1
    NEGATIVE_EXAMPLES = 2
    UNPROCESSED_EXAMPLES = 3
    INIT_CLAUSES = 4
    POST_PROCESSING = 5
    ERROR = 6


def test_line(line_text):
    chars = line_text.split(",")
    if chars[0] == '':
        return False
    if 'Got FunCode out of bounds' in chars[0]:
        print("[!] ERROR: "+line_text)
        return True
    try:
        clause = [int(i) for i in chars]
    except ValueError as e:
        print("[!] ERROR"+str(e))
        return True
    for fun_code in clause:
        if fun_code not in VOCAB_CODES:
            print("[!] ERROR: Found code that is not in vocabulary: "+str(fun_code)+" (see "+line_text+")")
            return True
    return False

running_threads = 0
wrong_vocab = 0
resource_out = 0


def test_line_with_response(text_to_test):
    global wrong_vocab
    global status
    global running_threads
    if test_line(text_to_test):
        wrong_vocab += 1
        status = FileStatus.ERROR
        print("Found error!")
    running_threads -= 1


for file in ALL_FILES:
    start_time = time.time()
    file_name = PREFIX + file.split("/")[-2].rsplit(".", 1)[0].replace(".", "-")
    print("Extracting "+file+" to "+file_name+"...")
    output_pos_file = os.path.join(OUTPUT_DIR, file_name+"_pos.txt")
    output_neg_file = os.path.join(OUTPUT_DIR, file_name+"_neg.txt")
    output_unp_file = os.path.join(OUTPUT_DIR, file_name+"_unp.txt")
    output_conj_file = os.path.join(OUTPUT_DIR, file_name+"_conj.txt")
    output_init_file = os.path.join(OUTPUT_DIR, file_name+"_init.txt")
    current_text = list()
    with open(file, 'r') as f:
        f_neg = open(output_neg_file, 'w')
        f_pos = open(output_pos_file, 'w')
        f_unp = open(output_unp_file, 'w')
        f_conj = open(output_conj_file, 'w')
        f_init = open(output_init_file, 'w')
        status = FileStatus.PRE_PROCESSING
        for line in f:
            text = line.split('\t')[-1].split('\n')[0]
            # print("Status "+str(status)+", Line "+text)
            if status == FileStatus.POST_PROCESSING:
                break
            elif status == FileStatus.POSITIVE_EXAMPLES:
                if "# Training: Positive examples end" not in text:
                    if text == "":
                        continue
                    #if text in current_text:
                    #    continue
                    if "Processing clause" in text:
                        text = text.split("Processing clause")[0]
                    current_text.append(text)
                    # running_threads += 1
                    # _thread.start_new_thread(test_line_with_response, (text,))
                else:
                    # while running_threads > 0:
                    #     print("Wait on thread with "+str(running_threads)+" running threads")
                    #     time.sleep(1)
                    # if status == FileStatus.ERROR:
                    #     break
                    with Pool(processes=12) as pool:
                        current_text = set(current_text)
                        current_text = list(current_text)
                        res = pool.map(test_line, current_text)
                        if any(res):
                            wrong_vocab += 1
                            status = FileStatus.ERROR
                            break
                        else:
                            s = "\n".join(current_text)
                            f_pos.write(s)
                            current_text = list()
                            status = FileStatus.PRE_PROCESSING

            elif status == FileStatus.NEGATIVE_EXAMPLES:
                if "# Training: Negative examples end" not in text:
                    if text == "":
                        continue
                    #if text in current_text:
                    #    continue
                    if "Processing clause" in text:
                        text = text.split("Processing clause")[0]
                    current_text.append(text)
                    # running_threads += 1
                    # _thread.start_new_thread(test_line_with_response, (text,))
                else:
                    with Pool(processes=12) as pool:
                        current_text = set(current_text)
                        current_text = list(current_text)
                        res = pool.map(test_line, current_text)
                        if any(res):
                            wrong_vocab += 1
                            status = FileStatus.ERROR
                            break
                        else:
                            s = "\n".join(current_text)
                            f_neg.write(s)
                            current_text = list()
                            # status = FileStatus.PRE_PROCESSING
                            status = FileStatus.POST_PROCESSING

            elif status == FileStatus.UNPROCESSED_EXAMPLES:
                if "# Training: Unprocessed examples end" not in text:
                    if text == "":
                        continue
                    #if text in current_text:
                    #    continue
                    if "Processing clause" in text:
                        text = text.split("Processing clause")[0]
                    current_text.append(text)
                    # running_threads += 1
                    # _thread.start_new_thread(test_line_with_response, (text,))
                else:
                    with Pool(processes=12) as pool:
                        current_text = set(current_text)
                        current_text = list(current_text)
                        res = pool.map(test_line, current_text)
                        if any(res):
                            wrong_vocab += 1
                            status = FileStatus.ERROR
                            break
                        else:
                            s = "\n".join(current_text)
                            f_unp.write(s)
                            current_text = list()
                            status = FileStatus.POST_PROCESSING

            elif status == FileStatus.INIT_CLAUSES:
                if "# Init clauses end" not in text:
                    f_init.write(text+"\n")
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
                elif "# Training: Unprocessed examples begin" in text:
                    status = FileStatus.UNPROCESSED_EXAMPLES
                elif "# Init clauses begin" in text:
                    status = FileStatus.INIT_CLAUSES
                elif "# Negative Conjecture:" in text or "# Conjecture:" in text:
                    conjecture = text.split(':')[-1]
                    f_conj.write(conjecture + "\n")
                    if test_line(conjecture):
                        wrong_vocab += 1
                        status = FileStatus.ERROR
                        break
                elif "EOF" in text:
                    print("Found "+text+" while pre-processing")
                    resource_out += 1
                    status = FileStatus.ERROR
                    break
            elif status == FileStatus.ERROR:
                break
        f_neg.close()
        f_pos.close()
        f_unp.close()
        f_conj.close()
        f_init.close()
        if status == FileStatus.ERROR:
            os.remove(output_init_file)
            os.remove(output_pos_file)
            os.remove(output_neg_file)
            os.remove(output_conj_file)
            os.remove(output_unp_file)
        print("Done in "+str(time.time()-start_time)+"sec.")
print("Found "+str(wrong_vocab)+" files with vocab problems")
print("Found "+str(resource_out)+" files with resource out")
