from glob import glob
from random import shuffle
from TPTP_train_val_files import get_TPTP_ClauseWeight_test


EXECUTION_LINE = './eprover --output-level=0 --cpu-limit=1800 --training-examples=3 --vocab-arity-file=DeepReasoning_Arity.txt --vocab-max=2048 --vocab-file=DeepReasoning_Vocabs.txt -H"(20.TensorFlowWeight(ConstPrio),1.ConjectureRelativeSymbolWeight(SimulateSOS,0.5,100,100,100,100,1.5,1.5,1),1.ConjectureRelativeSymbolWeight(ConstPrio,0.1,100,100,100,100,1.5,1.5,1.5),1.FIFOWeight(PreferProcessed),1.ConjectureRelativeSymbolWeight(PreferNonGoals,0.5,100,100,100,100,1.5,1.5,1),1.Refinedweight(SimulateSOS,3,2,2,1.5,2))"' # 1.Clauseweight(ConstPrio,2,1,1)
FILE_LIST = [sorted(glob("/home/phillip/datasets/TPTP_small/Problems/SET/*.p")),
             sorted(glob("/home/phillip/datasets/TPTP_small/Problems/REL/*.p")),
             sorted(glob("/home/phillip/datasets/TPTP_small/Problems/GRP/*.p")),
             sorted(glob("/home/phillip/datasets/TPTP_small/Problems/PRO/*.p")),
             sorted(glob("/home/phillip/datasets/TPTP_small/Problems/ROB/*.p")),
             sorted(glob("/home/phillip/datasets/TPTP_small/Problems/LDA/*.p"))]
print(FILE_LIST)
#FILE_LIST = ["/home/phillip/datasets/TPTP_small/Problems/"+f[:3]+"/"+f+".p" for f in get_TPTP_ClauseWeight_test()]
#shuffle(FILE_LIST)
OUTPUT_FOLDER = "/home/phillip/datasets/results/BestHeuristic/"
TRAINING_LIST = [f.split("/")[-1].split("_")[1] for f in sorted(glob("/home/phillip/datasets/Cluster/BestWithUnprocessed/*_pos.txt"))]

FILE_NAME = "eprover_run.sh"

with open(FILE_NAME, "w") as bash_file:
    all_lines = list()
    for domain_list in FILE_LIST:
        domain_lines = list()
        for problem_file in domain_list:
            if problem_file.split("/")[-1].split(".")[0] not in TRAINING_LIST and "^" not in problem_file:
                line = EXECUTION_LINE + " " + problem_file + " > " + OUTPUT_FOLDER + problem_file.split("/")[-1].rsplit(".",1)[0].replace(".","_") + "_output.txt\n"
                domain_lines.append(line)
                # bash_file.write(line)
        shuffle(domain_lines)
        all_lines.append(domain_lines)
    for index in range(max([len(l) for l in all_lines])):
        for domain_lines in all_lines:
            if index < len(domain_lines):
                bash_file.write(domain_lines[index])
