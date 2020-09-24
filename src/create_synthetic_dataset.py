from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
import os
import argparse
import json
import random
import sys

    
def create_hist(data_dir_prefix='./data/snli_1.0/cl_snli', model_name='bert-base-uncased'):
    hard_test_labels = []
    hard_test_lines = []

    with open(data_dir_prefix + '_test_lbl_file') as test_labels_file:
        test_labels = test_labels_file.readlines()
    with open(data_dir_prefix + '_test_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
    with open(data_dir_prefix + '_train_lbl_file') as train_labels_file:
        train_labels = train_labels_file.readlines()
    with open(data_dir_prefix + '_train_source_file') as train_lines_file:
        train_lines = train_lines_file.readlines()
    with open(data_dir_prefix + '_val_lbl_file') as val_labels_file:
        val_labels = val_labels_file.readlines()
    with open(data_dir_prefix + '_val_source_file') as val_lines_file:
        val_lines = val_lines_file.readlines()
    if os.path.isfile(data_dir_prefix + '_test_hard_lbl_file') and \
        os.path.isfile(data_dir_prefix + '_test_hard_source_file'):
        with open(data_dir_prefix + '_test_hard_lbl_file') as val_labels_file:
            hard_test_labels = val_labels_file.readlines()
        with open(data_dir_prefix + '_test_hard_source_file') as val_lines_file:
            hard_test_lines = val_lines_file.readlines()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    res = {'total':defaultdict(int), 'contradiction':defaultdict(int), 'entailment':defaultdict(int), 'neutral':defaultdict(int)}

    for label,line in zip(train_labels+val_labels+test_labels,train_lines+val_lines+test_lines):
        premise, hypothesis = line.split('|||')[0:2]
        label = label.strip()
        hypothesis_ids = tokenizer.encode(hypothesis)
        for hyp_id in hypothesis_ids:
            res['total'][hyp_id] += 1
            res[label][hyp_id] += 1

    np.save('hist_all.npy',res)

    return res


def parse_cli():
    p = argparse.ArgumentParser(description='Create syntatic dataset')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_exp = sp.add_parser('run-exp', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_experiment)
    sp_exp.add_argument('--run-name', '-n', type=str,
                        help='Name of run and output file', required=True)
    sp_exp.add_argument('--out-dir', '-o', type=str, help='Output folder',
                        default='./results', required=False)
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        default=None, required=False)
    sp_exp.add_argument('--drive', '-d', type=bool, help='Pass "True" if you are running this on Google Colab',
                        default=False, required=False)
    sp_exp.add_argument('--do-test', '-t', type=bool, help='Pass "True" if you want to run a test on test set',
                        default=True, required=False)                    

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed
        
    


if __name__ == "__main__":
    bias_token0 = 30000
    bias_token1 = 30001
    bias_token2 = 30002
    create_hist()
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
