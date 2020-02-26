import argparse
import json
import os
import random
import sys

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from src.data import PremiseGenerationDataset
from src.models import get_model
from src.train import PremiseGeneratorTrainer
from src.utils import FitResult, get_max_len
import math


def run_experiment(run_name, out_dir='./results', data_dir_prefix='./data/snli_1.0/cl_snli',
                   model_name='bert-base-uncased', seed=None,
                   # Training params
                   bs_train=32, bs_test=None, batches=100, epochs=100,
                   early_stopping=3, checkpoints=None, lr=0.0005, reg=1e-3, max_len=0,
                   **kw):
    """
    :param max_len:
    :param model_name:
    :param data_dir_prefix:
    :param run_name:
    :param out_dir:
    :param seed:
    :param bs_train:
    :param bs_test:
    :param batches:
    :param epochs:
    :param early_stopping:
    :param checkpoints:
    :param lr:
    :param reg:
    :param kw:
    :return:
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()

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

    tokenizer = BertTokenizer.from_pretrained(model_name)
    if max_len == 0:
        max_len = get_max_len(test_lines[:batches*bs_test] + train_lines[:batches*bs_train] + val_lines[:batches*bs_test],
                              '|||', tokenizer)
        print(f'Longest Sequence is: {max_len} token_ids')

    max_len = math.pow(2, math.ceil(math.log(max_len, 2)))
    max_len = int(max_len)

    print(f'Setting max_len to: {max_len}')

    all_labels_text = list(set(test_labels[:batches*bs_test] + train_labels[:batches*bs_train] + val_labels[:batches*bs_test]))

    all_labels = ['['+l.upper().replace('\n', '')+']' for l in all_labels_text]

    tokenizer.add_tokens(all_labels)

    labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]

    print(f'Labels IDs: {labels_ids}')

    ds_test = PremiseGenerationDataset(test_lines, test_labels, tokenizer, max_len=max_len)
    ds_val = PremiseGenerationDataset(val_lines, val_labels, tokenizer, max_len=max_len)
    ds_train = PremiseGenerationDataset(train_lines, train_labels, tokenizer, max_len=max_len)

    ds_test = Subset(ds_test, range(batches*bs_test))
    ds_val = Subset(ds_val, range(batches*bs_test))
    ds_train = Subset(ds_train, range(batches*bs_train))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(tokenizer=tokenizer, model='encode-decode')

    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=False)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_test, shuffle=False)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False)
    # print(model)
    optimizer = AdamW(model.parameters(), lr=lr)

    trainer = PremiseGeneratorTrainer(model, tokenizer, None, optimizer, max_len, labels_ids, device)
    fit_res = trainer.fit(dl_train, dl_val, num_epochs=epochs, early_stopping=early_stopping)
    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, config, fit_res):
    output = dict(
        config=config,
        results=fit_res._asdict()
    )
    output_filename = f'{os.path.join(out_dir, run_name)}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')


def load_experiment(filename):
    with open(filename, 'r') as f:
        output = json.load(f)

    config = output['config']
    fit_res = FitResult(**output['results'])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description='Experiments')
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

    # # Training
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-test', type=int, help='Test batch size',
                        metavar='BATCH_SIZE')
    sp_exp.add_argument('--batches', type=int,
                        help='Number of batches per epoch', default=100)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of epochs', default=100)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without '
                             'improvement', default=3)
    sp_exp.add_argument('--checkpoints', type=int,
                        help='Save model checkpoints to this file when test '
                             'accuracy improves', default=None)
    sp_exp.add_argument('--lr', type=float,
                        help='Learning rate', default=0.001)
    sp_exp.add_argument('--reg', type=float,
                        help='L2 regularization', default=1e-3)
    sp_exp.add_argument('--data-dir-prefix', type=str,
                        help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_exp.add_argument('--max-len', type=int,
                        help='Length of longest sequence (or bigger), 0 if you don\'t know', default=0)

    # # Model
    sp_exp.add_argument('--model-name', type=str,
                        help='Name of the huggingface model', default='bert-base-uncased')
    # sp_exp.add_argument('--filters-per-layer', '-K', type=int, nargs='+',
    #                     help='Number of filters per conv layer in a block',
    #                     metavar='K', required=True)
    # sp_exp.add_argument('--layers-per-block', '-L', type=int, metavar='L',
    #                     help='Number of layers in each block', required=True)
    # sp_exp.add_argument('--pool-every', '-P', type=int, metavar='P',
    #                     help='Pool after this number of conv layers',
    #                     required=True)
    # sp_exp.add_argument('--hidden-dims', '-H', type=int, nargs='+',
    #                     help='Output size of hidden linear layers',
    #                     metavar='H', required=True)
    # sp_exp.add_argument('--ycn', action='store_true', default=False,
    #                     help='Whether to use your custom network')

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed


# run_experiment("dimi", data_dir_prefix="../data/scitail/cl_scitail", bs_train=8, bs_test=4)

if __name__ == '__main__':
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
