import argparse
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

label2idx = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--biased_preds")
    parser.add_argument("--model_preds")
    parser.add_argument("--lbl_file")
    #parser.add_argument("--alpha", type=float, default=0.1)
    #parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--combine_nonentailments", action='store_true')

    args = parser.parse_args()
    
    with open(args.biased_preds, 'r') as f:
        next(f) # Take off first line
        biased_preds = {}
        for line in f:
            parts = line.strip().split(',')
            index = int(parts[0])
            pred = parts[1]
            biased_preds[index] = pred

    with open(args.model_preds, 'r') as f:
        next(f) # Take off first line
        model_preds = {}
        for line in f:
            parts = line.strip().split(',')
            index = int(parts[0])
            pred = parts[1]
            model_preds[index] = pred

    with open(args.lbl_file, 'r') as f:
        labels = []
        for line in f:
            labels.append(line.strip())
        labels = np.array(labels)

    biased_keys = list(sorted(biased_preds.keys()))
    model_keys = list(sorted(model_preds.keys()))

    print(len(biased_keys), len(model_keys), len(labels))
    print(biased_keys == model_keys)

    keys = biased_keys

    biased_preds = np.array([biased_preds[k] for k in keys])
    model_preds = np.array([model_preds[k] for k in keys])
    
    labels_hard = labels[biased_preds != labels]
    biased_preds_hard = biased_preds[biased_preds != labels]
    model_preds_hard = model_preds[biased_preds != labels]

    labels_easy = labels[biased_preds == labels]
    biased_preds_easy = biased_preds[biased_preds == labels]
    model_preds_easy = model_preds[biased_preds == labels]

    biased_acc = (biased_preds == labels).mean()
    model_acc = (model_preds == labels).mean()

    biased_hard_acc = (biased_preds_hard == labels_hard).mean()
    model_hard_acc = (model_preds_hard == labels_hard).mean()

    biased_easy_acc = (biased_preds_easy == labels_easy).mean()
    model_easy_acc = (model_preds_easy == labels_easy).mean()

    print(f'Full: Biased acc = {biased_acc}, Model acc = {model_acc}')
    print(f'Hard: Biased acc = {biased_hard_acc}, Model acc = {model_hard_acc}')
    print(f'Easy: Biased acc = {biased_easy_acc}, Model acc = {model_easy_acc}')


if __name__ == '__main__':
    main()
