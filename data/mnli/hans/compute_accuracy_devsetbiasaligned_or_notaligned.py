import numpy as np
import sys

def acc(p, g, start, end):
    return (p[start:end] == g[start:end]).mean()

gt_path = 'hans_evalset_full_lbl_file'

gt_labels = []
with open(gt_path, 'r') as f:
    for line in f:
        gt_labels.append(line.strip())
gt_labels = np.array(gt_labels)

results_path = sys.argv[1] #'../../../results/hans_test_results_hansevalset.csv'

preds = [None for _ in range(len(gt_labels))]
with open(results_path, 'r') as f:
    next(f)
    for line in f:
        idx, pred = line.strip().split(',')
        preds[int(idx)] = pred
preds = np.array(preds)

acc_lex_nonent = acc(preds, gt_labels, 0, 5000)
acc_lex_ent = acc(preds, gt_labels, 5000, 10000)
acc_subseq_nonent = acc(preds, gt_labels, 10000, 15000)
acc_subseq_ent = acc(preds, gt_labels, 15000, 20000)
acc_const_nonent = acc(preds, gt_labels, 20000, 25000)
acc_const_ent = acc(preds, gt_labels, 25000, 30000)

print(acc_lex_nonent, acc_lex_ent, acc_subseq_nonent, acc_subseq_ent, acc_const_nonent, acc_const_ent)
