import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression

idx2label = ["contradiction", "entailment", "neutral"]
label2idx = {"contradiction": 0, "entailment": 1, "neutral": 2}

def frac_overlap(p, h):
    p = p.split(' ')
    h = h.split(' ')

    total_overlap = 0
    for h_i in h:
        if h_i in p:
            total_overlap += 1
    return total_overlap/len(h)

def load_data(prefix):
    source_file = prefix+'_source_file'
    label_file = prefix+'_lbl_file'
    id_file = prefix+'_id_file'

    premises = []
    hypotheses = []
    labels = []
    ids = []
    fraction_overlap = []

    with open(source_file, 'r') as f:
        for line in f:
            p, h = line.strip().split('|||')
            premises.append(p)
            hypotheses.append(h)
            fraction_overlap.append(frac_overlap(p, h))

    with open(label_file, 'r') as f:
        for line in f:
            label = line.strip()
            labels.append(label2idx[label])
    
    if os.path.exists(id_file):
        with open(id_file, 'r') as f:
            for line in f:
                id_i = line.strip()
                ids.append(id_i)
    else:
        ids = None

    X = np.array(fraction_overlap)[:, np.newaxis]
    y = np.array(labels)
    return X, y, ids

def train():
    X, y, _ = load_data('../cl_multinli_train')
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf, clf.score(X, y)

def dev_acc(clf):
    X, y, _ = load_data('../cl_multinli_dev_matched')
    return clf.score(X, y)

def hans_acc(clf):
    X, y, _ = load_data('hans_evalset_lexical_overlap')
    lexical_overlap_acc = clf.score(X, y)

    X, y, _ = load_data('hans_evalset_subsequence')
    subsequence_acc = clf.score(X, y)

    X, y, _ = load_data('hans_evalset_constituent')
    constituent_acc = clf.score(X, y)

    return lexical_overlap_acc, subsequence_acc, constituent_acc

def hans_predictions(clf):
    X1, _, ids1 = load_data('hans_evalset_lexical_overlap')
    X2, _, ids2 = load_data('hans_evalset_subsequence')
    X3, _, ids3 = load_data('hans_evalset_constituent')
    X = np.concatenate([X1, X2, X3], 0)
    ids = np.concatenate([ids1, ids2, ids3], 0)

    logprobs = clf.predict_log_proba(X)
    logprobs_dict = {id_i: logprob_i.tolist() for id_i, logprob_i in zip(ids, logprobs)}

    return logprobs_dict

if __name__ == '__main__':
    clf, train_acc = train()

    logprobs_dict = hans_predictions(clf)
    print(len(logprobs_dict))
    with open(f"output/wordoverlapmodel_logprobs_allsubsets.json", 'w') as f:
        json.dump(logprobs_dict, f)

    #print(train_acc)
    #print(dev_acc(clf))
    #print(hans_acc(clf))
