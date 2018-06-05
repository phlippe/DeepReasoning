import os
import sys
import numpy as np
from glob import glob


def main(test_files):
    best_acc_pos = 0
    best_acc_neg = 0
    best_file = "None"
    for file in test_files:
        # if not int(file.split("_")[-1].split(".")[0]) % 2000 == 0:
        #     continue
        print("="*40)
        print(file)
        print("="*40)
        results = dict()
        with open(file, 'r') as f:
            for line in f:
                proof_name = line.split("->")[0].replace(" ", "")
                if proof_name not in results:
                    results[proof_name] = dict()
                    results[proof_name][0] = list()
                    results[proof_name][1] = list()
                numbers = line.split("Weight:")[-1].split("|")[0].replace(" ","").split("/")
                results[proof_name][int(numbers[1])].append(float(numbers[0]))

        loc_acc_pos_list = None
        loc_acc_neg_list = None
        loc_mean_acc_pos = 0.0
        loc_mean_acc_neg = 0.0
        loc_margin = 0.0
        num_pos = 0
        num_neg = 0
        for mfactor in range(50, 51):
            margin = mfactor / 100.0
            acc_pos_list = list()
            acc_neg_list = list()
            for key, value in results.iteritems():
                num_pos = num_pos + len(value[0])
                num_neg = num_neg + len(value[1])
                acc_pos = len([v for v in value[0] if v <= margin]) * 100.0 / len(value[0])
                acc_neg = len([v for v in value[1] if v >= margin]) * 100.0 / len(value[1])
                acc_pos_list.append(acc_pos)
                acc_neg_list.append(acc_neg)
            print("Overall positive: "+str(num_pos)+", negative: "+str(num_neg))
            all_positives = [item for sublist in [value[0] for key, value in results.iteritems()] for item in sublist]
            all_negatives = [item for sublist in [value[1] for key, value in results.iteritems()] for item in sublist]
            mean_acc_pos = np.mean(np.array(acc_pos_list)) # len([v for v in all_positives if v <= margin]) * 100.0 / len(all_positives)
            mean_acc_neg = np.mean(np.array(acc_neg_list)) # len([v for v in all_negatives if v >= margin]) * 100.0 / len(all_negatives)
            if (mean_acc_pos + mean_acc_neg) > (loc_mean_acc_pos + loc_mean_acc_neg):
                loc_acc_pos_list = acc_pos_list
                loc_acc_neg_list = acc_neg_list
                loc_mean_acc_pos = mean_acc_pos
                loc_mean_acc_neg = mean_acc_neg
                loc_margin = margin

        index = 0
        for key, value in results.iteritems():
            print(key+":\t"+("%5.2f" % loc_acc_pos_list[index])+"\t"+("%5.2f" % loc_acc_neg_list[index]))
            index += 1
        print("-"*40)

        print("Total:\t\t"+("%.2f" % loc_mean_acc_pos) + "\t"+("%.2f" % loc_mean_acc_neg)+"\tMargin: "+str(loc_margin))
        if (loc_mean_acc_pos + loc_mean_acc_neg) > (best_acc_pos + best_acc_neg):
            best_file = file
            best_acc_pos = loc_mean_acc_pos
            best_acc_neg = loc_mean_acc_neg

    print("\n"+("#"*40+"\n")*3)
    print("Best file: "+best_file)
    print("Accuracy positive: "+str(best_acc_pos))
    print("Accuracy negative: "+str(best_acc_neg))



if __name__ == '__main__':
    main(sorted(glob("logs/test/2018_05_28*/test_file_*.txt"), key=lambda x: int(x.split("_")[-1].split(".")[0])))