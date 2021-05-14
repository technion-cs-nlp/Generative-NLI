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
    
    model_correctness = (model_preds == labels).astype(np.float)
    biased_correctness = (biased_preds == labels).astype(np.float)

    correlation = np.corrcoef(biased_correctness, model_correctness)
    print(correlation)

if __name__ == '__main__':
    main()
