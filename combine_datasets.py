from glob import glob
import os

FILES_CW = sorted(glob("/home/phillip/datasets/Cluster/Training/Clau*_neg.txt"))
FILES_BEST_PREFIX = "/home/phillip/datasets/Cluster/BestHeuristic/"
NEW_FILES_PREFIX = "/home/phillip/datasets/Cluster/Combined/"

for f_neg in FILES_CW:
    f_pos = f_neg.replace("_neg", "_pos")
    best_neg = FILES_BEST_PREFIX + f_neg.replace("ClauseWeight", "Best").split("/")[-1]
    new_neg = NEW_FILES_PREFIX + f_neg.replace("ClauseWeight", "Combined").split("/")[-1]
    print("Files "+f_neg+" combine with "+best_neg+" to "+new_neg)
    with open(f_neg, "r") as file_clause_weight_neg:
        with open(f_pos, "r") as file_clause_weight_pos:
            with open(new_neg, "w") as file_new_neg:
                for line in file_clause_weight_neg:
                    file_new_neg.write(line)
                if os.path.exists(best_neg):
                    with open(best_neg, "r") as file_best_neg:
                        for line in file_best_neg:
                            if line not in file_clause_weight_neg and line not in file_clause_weight_pos:
                                file_new_neg.write(line)