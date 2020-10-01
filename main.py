import argparse
import json
import os
import random
import sys

import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AdamW, AutoModel, \
get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup

from src.data import PremiseGenerationDataset, DiscriminativeDataset, HypothesisOnlyDataset, DualDataset
from src.models import get_model, HybridModel
from src.train import PremiseGeneratorTrainer, DiscriminativeTrainer, OnelabelTrainer, HybridTrainer
from src.utils import FitResult, get_max_len
import math
from torch.utils.tensorboard import SummaryWriter
from torch.optim.sgd import SGD


def run_experiment(run_name, out_dir='./results', data_dir_prefix='./data/snli_1.0/cl_snli', model_path=None,
                   model_name='bert-base-uncased', model_type='encode-decode', decoder_model_name=None, seed=None,
                   drive=False, do_test=True,
                   # Training params
                   bs_train=32, bs_test=None, batches=0, epochs=20,
                   early_stopping=3, checkpoints=None, lr=0.0005, reg=1e-3, max_len=0, decoder_max_len=0,
                   optimizer_type='Adam', momentum=0.9, word_dropout=0.0,label_smoothing_epsilon=0.0,
                   tie_embeddings=False, hypothesis_only=False, generate_hypothesis=False,
                   # Model params
                   beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.0, param_freezing_ratio=0.0, gradual_unfreeze=False,
                   ret_res=False, gamma=0.5, 
                   # Dataset params
                   inject_bias=0, bias_ids=30000, bias_ratio=0.5, bias_location='start', non_discriminative_bias=False,
                   label=None,
                   **kw):
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()

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
    if 'gpt' in model_name:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer_decoder = None
    if decoder_model_name is not None and 'gpt' in decoder_model_name:
        tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_model_name)
        tokenizer_decoder.pad_token = tokenizer_decoder.eos_token

    size_test = batches * bs_test if batches > 0 else 10**8
    size_train = batches * bs_train if batches > 0 else 10**8

    all_labels_text = list(set(test_labels[:size_test] + train_labels[:size_train] + val_labels[:size_test] + hard_test_labels[:size_test]))
    all_labels_text.sort()
    num_labels = len(all_labels_text)

    if label is not None:
        if model_type.startswith('disc'):
            raise AttributeError("Can't specify label with discriminative model")
        train_lines = np.array(train_lines)[np.array(train_labels)==all_labels_text[label]].tolist()
        train_labels = np.array(train_labels)[np.array(train_labels)==all_labels_text[label]].tolist()
        val_lines = np.array(val_lines)[np.array(val_labels)==all_labels_text[label]].tolist()
        val_labels = np.array(val_labels)[np.array(val_labels)==all_labels_text[label]].tolist()
        test_lines = np.array(test_lines)[np.array(test_labels)==all_labels_text[label]].tolist()
        test_labels = np.array(test_labels)[np.array(test_labels)==all_labels_text[label]].tolist()

    # import pdb; pdb.set_trace()

    if model_type != 'discriminative':

        all_labels = ['[' + l.upper().replace('\n', '') + ']' for l in all_labels_text]

        tokenizer.add_tokens(all_labels)
        labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]
        print(f'Labels IDs: {labels_ids}')
        # if tokenizer_decoder is not None:
        #     tokenizer_decoder.add_tokens(all_labels)
        #     labels_ids_decoder = [tokenizer_decoder.encode(label, add_special_tokens=False)[0] for label in all_labels]
        #     print(f'Labels IDs for decoder: {labels_ids_decoder}')

    dataset = None
    trainer_type = None
    data_args = {}
    dataloader_args = {}
    train_args = {}
    # import pdb; pdb.set_trace()
    if model_type in ['encode-decode','bart','shared']:
        dataset = DiscriminativeDataset
        if label is None:
            trainer_type = PremiseGeneratorTrainer
        else:
            trainer_type = OnelabelTrainer
            train_args['label'] = label
        train_args['possible_labels_ids'] = labels_ids
        train_args['epsilon'] = label_smoothing_epsilon
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['gradual_unfreeze'] = gradual_unfreeze
        # dataloader_args['collate_fn'] = my_collate

    elif model_type == 'discriminative':
        if hypothesis_only:
            dataset = HypothesisOnlyDataset
        else:
            dataset = DiscriminativeDataset
        trainer_type = DiscriminativeTrainer
        train_args['num_labels'] = num_labels
        train_args['tokenizer'] = tokenizer

    elif model_type == 'hybrid':
        dataset = DiscriminativeDataset
        trainer_type = HybridTrainer
        train_args['possible_labels_ids'] = labels_ids
        train_args['epsilon'] = label_smoothing_epsilon
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['gradual_unfreeze'] = gradual_unfreeze
        train_args['num_labels'] = num_labels

    ds_test = dataset(test_lines, test_labels, tokenizer, max_len=max_len, **data_args)
    ds_val = dataset(val_lines, val_labels, tokenizer, max_len=max_len, **data_args)
    # data_args['dropout'] = word_dropout
    data_args.update({
        'inject_bias':inject_bias,
        'bias_ids':bias_ids,
        'bias_ratio':bias_ratio,
        'bias_location':bias_location,
        'non_discriminative_bias': non_discriminative_bias,
        'dropout':word_dropout,
    })
    ds_train = dataset(train_lines, train_labels, tokenizer, max_len=max_len, **data_args)

    if batches > 0:
        # ds_test = Subset(ds_test, range(batches * bs_test))
        # ds_val = Subset(ds_val, range(batches * bs_test))
        ds_train = Subset(ds_train, range(batches * bs_train))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(  tokenizer=tokenizer, 
                        tokenizer_decoder=tokenizer_decoder, 
                        model=model_type, 
                        model_name=model_name,
                        decoder_model_name=decoder_model_name, 
                        model_path=model_path, 
                        param_freezing_ratio=param_freezing_ratio,
                        tie_embeddings=tie_embeddings,
                        label=label,
                        gamma=gamma)

    model.to(device)
    # import pdb; pdb.set_trace()
    
    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=True, **dataloader_args)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_test, shuffle=False, **dataloader_args)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False, **dataloader_args)
    
    if optimizer_type.lower() == 'adam':
        optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise AttributeError('only SGD and Adam supported for now')
    
    num_batches = batches if batches > 0 else len(dl_train)
    num_steps = epochs * num_batches
    print(f"Number of training steps: {num_steps}")
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2*num_batches, num_training_steps=num_steps)
    writer = SummaryWriter()
    # import pdb; pdb.set_trace()
    trainer = trainer_type(model, optimizer, scheduler, max_len=max_len, device=device, **train_args)
    fit_res = trainer.fit(dl_train, dl_val, num_epochs=epochs, early_stopping=early_stopping, checkpoints=checkpoints,
                          drive=drive, writer=writer)

    if ret_res:
        del model
        return fit_res.test_acc[-4] if len(fit_res.test_acc) >=4 else fit_res.test_acc[-1]
    save_experiment(run_name, out_dir, cfg, fit_res)

    if do_test:
        trainer.test(dl_test,writer=writer)
        if len(hard_test_labels) > 0 and len(hard_test_lines) > 0:
            ds_hard_test = dataset(hard_test_lines, hard_test_labels, tokenizer, max_len=max_len, **data_args)
            if batches > 0:
                ds_test = Subset(ds_hard_test, range(batches * bs_test))
            dl_hard_test = torch.utils.data.DataLoader(ds_hard_test, bs_test, shuffle=False)
            trainer.test(dl_hard_test, writer=writer)


def test_model(run_name, out_dir='./results_test', data_dir_prefix='./data/snli_1.0/cl_snli',
               model_name='bert-base-uncased', model_path=None, model_type='encode-decode', decoder_model_name=None, seed=None, save_results=None,
               # Training params
               bs_test=None, batches=0,
               checkpoints=None, max_len=0, decoder_max_len=0, 
               hypothesis_only=False, generate_hypothesis=False, create_premises=False, label=0,
               **kw):
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = 1
    cfg = locals()

    tf = torchvision.transforms.ToTensor()

    hard_test_labels = None
    hard_test_lines = None

    with open(data_dir_prefix + '_test_lbl_file') as test_labels_file:
        test_labels = test_labels_file.readlines()
    with open(data_dir_prefix + '_test_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
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
    # import pdb; pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'gpt' in model_name:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer_decoder = None
    if decoder_model_name is not None and 'gpt' in decoder_model_name:
        tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_model_name)
        tokenizer_decoder.pad_token = tokenizer_decoder.unk_token

    size_test = batches * bs_test if batches > 0 else 10**8

    all_labels_text = list(set(test_labels[:size_test] + val_labels[:size_test] + hard_test_labels[:size_test]))
    all_labels_text.sort()
    num_labels = len(all_labels_text)

    if model_type != 'discriminative':

        all_labels = ['[' + l.upper().replace('\n', '') + ']' for l in all_labels_text]

        tokenizer.add_tokens(all_labels)
        labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]
        print(f'Labels IDs: {labels_ids}')
        # if tokenizer_decoder is not None:
        #     tokenizer_decoder.add_tokens(all_labels)
        #     labels_ids_decoder = [tokenizer_decoder.encode(label, add_special_tokens=False)[0] for label in all_labels]
        #     print(f'Labels IDs for decoder: {labels_ids_decoder}')

    dataset = None
    trainer_type = None
    data_args = {}
    dataloader_args = {}
    train_args = {}
    # import pdb; pdb.set_trace()
    if model_type in ['encode-decode','bart','shared']:
        dataset = DiscriminativeDataset
        if label is None:
            trainer_type = PremiseGeneratorTrainer
        else:
            trainer_type = OnelabelTrainer
        # data_args['tokenizer_decoder'] = tokenizer_decoder
        # data_args['generate_hypothesis'] = generate_hypothesis
        train_args['possible_labels_ids'] = labels_ids
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['create_premises'] = create_premises
        train_args['save_results'] = save_results
        # dataloader_args['collate_fn'] = my_collate
    elif model_type == 'discriminative':
        if hypothesis_only:
            dataset = HypothesisOnlyDataset
        else:
            dataset = DiscriminativeDataset
        trainer_type = DiscriminativeTrainer
        train_args['num_labels'] = num_labels
        train_args['tokenizer'] = tokenizer

    # import pdb; pdb.set_trace()

    ds_test = dataset(test_lines, test_labels, tokenizer, max_len=max_len, **data_args)
    ds_val = dataset(val_lines, val_labels, tokenizer, max_len=max_len, **data_args)

    if batches > 0:
        ds_test = Subset(ds_test, range(batches * bs_test))
        ds_val = Subset(ds_val, range(batches * bs_test))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(tokenizer=tokenizer, tokenizer_decoder=tokenizer_decoder, model=model_type, model_name=model_name,
                            decoder_model_name=decoder_model_name, model_path=model_path)

    model.to(device)
                            
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False, **dataloader_args)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_test, shuffle=False, **dataloader_args)

    writer = None
    if checkpoints is None:
        writer = SummaryWriter()
    
    trainer = trainer_type(model, optimizer=None, scheduler=None, max_len=max_len, device=device, **train_args)
    fit_res = trainer.test(dl_test,checkpoints=checkpoints, writer=writer)
    save_experiment(run_name, out_dir, cfg, fit_res)
    if hard_test_labels is not None and hard_test_lines is not None:
            if trainer.save_results is not None:
                trainer.save_results += '_hard'
            ds_hard_test = dataset(hard_test_lines, hard_test_labels, tokenizer, max_len=max_len, **data_args)
            if batches > 0:
                ds_test = Subset(ds_hard_test, range(batches * bs_test))
            dl_hard_test = torch.utils.data.DataLoader(ds_hard_test, bs_test, shuffle=False, **dataloader_args)
            fit_res = trainer.test(dl_hard_test, checkpoints=checkpoints, writer=writer)
            save_experiment(run_name + '_hard', out_dir, cfg, fit_res)


def combine_models( data_dir_prefix='./data/snli_1.0/cl_snli',bs_test=8,
                    modelA_path=None, modelA_type=None, 
                    modelA_name=None, modelB_path=None, 
                    modelB_type=None, modelB_name=None, 
                    gamma=0.5,hypothesis_only=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hard_test_labels = None
    hard_test_lines = None

    with open(data_dir_prefix + '_test_lbl_file') as test_labels_file:
        test_labels = test_labels_file.readlines()
    with open(data_dir_prefix + '_test_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
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

    if modelB_type is None:
        modelB_type = modelA_type
    if modelB_name is None:
        modelB_name = modelA_name

    modelA = get_model(model=modelA_type, model_name=modelA_name, model_path=modelA_path)
    modelB = get_model(model=modelB_type, model_name=modelB_name, model_path=modelB_path)
    model = HybridModel(modelA,modelB,gamma)

    model.to(device)

    tokenizerA = AutoTokenizer.from_pretrained(modelA_name)

    size_test = 10**8

    all_labels_text = list(set(test_labels[:size_test] + val_labels[:size_test] + hard_test_labels[:size_test]))
    all_labels_text.sort()
    num_labels = len(all_labels_text)
    all_labels = ['[' + l.upper().replace('\n', '') + ']' for l in all_labels_text]

    tokenizerA.add_tokens(all_labels)
    labels_ids = [tokenizerA.encode(label, add_special_tokens=False)[0] for label in all_labels]
    print(f'Labels IDs: {labels_ids}')

    if modelA_name == modelB_name:
        tokenizerB = tokenizerA
    
    else:
        tokenizerB = AutoTokenizer.from_pretrained(modelB_name)
        tokenizerB.add_tokens(all_labels)
        labels_ids_decoder = [tokenizerB.encode(label, add_special_tokens=False)[0] for label in all_labels]
        print(f'Labels IDs for second model: {labels_ids_decoder}')

    datasetA = PremiseGenerationDataset
    DatasetA = datasetA(test_lines,test_labels,tokenizerA)

    datasetB = HypothesisOnlyDataset if hypothesis_only else DiscriminativeDataset
    DatasetB = datasetB(test_lines,test_labels,tokenizerB)

    dataset = DualDataset(DatasetA,DatasetB)
    dataloader = torch.utils.data.DataLoader(dataset, bs_test, shuffle=False)

    trainer = DualTesterTrainer(model, tokenizer=tokenizerB, optimizer=None, scheduler=None, max_len=128, num_labels=3, possible_labels_ids=labels_ids, device=device)
    trainer.test(dataloader)

    if hard_test_labels is not None and hard_test_lines is not None:
            ds_hard_testA = datasetA(hard_test_lines, hard_test_labels, tokenizerA)
            ds_hard_testB = datasetB(hard_test_lines, hard_test_labels, tokenizerB)

            ds_hard_test = DualDataset(ds_hard_testA, ds_hard_testB)
            
            dl_hard_test = torch.utils.data.DataLoader(ds_hard_test, bs_test, shuffle=False)
            trainer.test(dl_hard_test)


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
    sp_exp.add_argument('--drive', '-d', type=bool, help='Pass "True" if you are running this on Google Colab',
                        default=False, required=False)
    sp_exp.add_argument('--do-test', '-t', type=bool, help='Pass "True" if you want to run a test on test set',
                        default=True, required=False)   
    # # Dataset
    #inject_bias=0, bias_ids=30000, bias_ratio=0.5, bias_location='start', seed=42                 
    sp_exp.add_argument('--inject-bias', type=int, help='Select number of labels to inject bias to their corresponding hypotheses',
                        default=0)
    sp_exp.add_argument('--bias-ids', type=list, help='Select the id of the biases symbols',
                        default=[2870,2874,2876])
    sp_exp.add_argument('--bias-ratio', type=float, help='Select the percentege of labels to inject bias to their corresponding hypotheses',
                        default=0.5)
    sp_exp.add_argument('--bias-location', type=str, help='Select where in the hypotheses to inject the bias, can be either "start" or "end", otherwise will be random location',
                        default='start')
    sp_exp.add_argument('--non-discriminative-bias','-ndb', dest='non_discriminative_bias', action='store_true')
    
    # # Training
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=128, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-test', type=int, help='Test batch size',
                        metavar='BATCH_SIZE')
    sp_exp.add_argument('--batches', type=int,
                        help='Number of batches per epoch', default=0)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of epochs', default=20)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without '
                             'improvement', default=3)
    sp_exp.add_argument('--checkpoints', type=str,
                        help='Save model checkpoints to this file when test '
                             'accuracy improves', default=None)
    sp_exp.add_argument('--lr', '-lr', type=float,
                        help='Learning rate', default=1e-5)
    sp_exp.add_argument('--reg', type=float,
                        help='L2 regularization', default=1e-3)
    sp_exp.add_argument('--data-dir-prefix', type=str,
                        help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_exp.add_argument('--max-len', '-ml', type=int,
                        help='Length of longest sequence (or bigger), 0 if you don\'t know', default=0)
    sp_exp.add_argument('--decoder-max-len', '-dml', type=int,
                        help='Length of longest sequence of the decoder (or bigger), 0 if you don\'t know', default=0)
    sp_exp.add_argument('--param-freezing-ratio', type=float,
                        help='How many of the params to freeze', default=0.0)
    sp_exp.add_argument('--optimizer-type', '-ot', type=str,
                        help='Which type of optimizer to use', default="Adam")
    sp_exp.add_argument('--momentum', '-m', type=float,
                        help='Momentum for SGD', default=0.9)
    sp_exp.add_argument('--word-dropout', '-wdo', type=float,
                        help='Word dropout rate during training', default=0.0)
    sp_exp.add_argument('--label-smoothing-epsilon', '-lse', type=float,
                        help='Epsilon argument for label smoothing (does not uses labels smoothing by \
                        default', default=0.0)
    sp_exp.add_argument('--tie-embeddings','-te', dest='tie_embeddings', action='store_true')
    sp_exp.add_argument('--hypothesis-only','-ho', dest='hypothesis_only', action='store_true')
    sp_exp.add_argument('--gradual-unfreeze','-gu', dest='gradual_unfreeze', action='store_true')
    sp_exp.add_argument('--generate-hypothesis','-gh', dest='generate_hypothesis', action='store_true')
    sp_exp.add_argument('--label', '-l', type=int,
                        help='Create generative model only for one label', default=None)
    sp_exp.set_defaults(tie_embeddings=False, hypothesis_only=False, generate_hypothesis=False, non_discriminative_bias=False, gradual_unfreeze=False)

    # # Model
    sp_exp.add_argument('--model-path', type=str,
                        help='Path to fined-tuned model', default=None)
    sp_exp.add_argument('--model-name', type=str,
                        help='Name of the huggingface model', default='bert-base-uncased')
    sp_exp.add_argument('--model-type', type=str,
                        help='Type of the model (encode-decode or hybrid)', default='encode-decode')
    sp_exp.add_argument('--decoder-model-name', type=str,
                        help='Name of the decoder, if empty then same as encoder', default=None)
    sp_exp.add_argument('--beta1', '-b1', type=float,
                        default=0.9)
    sp_exp.add_argument('--beta2', '-b2', type=float,
                        default=0.999)
    sp_exp.add_argument('--epsilon', '-eps', type=float,
                        default=1e-6)
    sp_exp.add_argument('--weight-decay', '-wd', type=float,
                        default=0.0)
    sp_exp.add_argument('--gamma', type=float,
                        default=0.5)
    # sp_exp.add_argument('--hidden-dims', '-H', type=int, nargs='+',
    #                     help='Output size of hidden linear layers',
    #                     metavar='H', required=True)
    # sp_exp.add_argument('--ycn', action='store_true', default=False,
    #                     help='Whether to use your custom network')

    # TEST
    sp_test = sp.add_parser('test', help='Test model on test set')
    sp_test.set_defaults(subcmd_fn=test_model)
    sp_test.add_argument('--run-name', '-n', type=str,
                         help='Name of run and output file', required=True)
    sp_test.add_argument('--out-dir', '-o', type=str, help='Output folder',
                         default='./results_test', required=False)
    sp_test.add_argument('--seed', '-s', type=int, help='Random seed',
                         default=None, required=False)
    sp_test.add_argument('--drive', '-d', type=bool, help='Pass "True" if you are running this on Google Colab',
                         default=False, required=False)
    sp_test.add_argument('--save-results', '-sr', type=str, help='Pass path if you want to save the results',
                         default=None, required=False)

    # # Inference
    sp_test.add_argument('--bs-test', type=int, help='Test batch size',
                         metavar='BATCH_SIZE')
    sp_test.add_argument('--batches', type=int,
                         help='Number of batches per epoch, pass "0" if you want the full database', default=0)
    sp_test.add_argument('--data-dir-prefix', type=str,
                         help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_test.add_argument('--max-len', type=int,
                         help='Length of longest sequence (or bigger), 0 if you don\'t know', default=0)
    sp_test.add_argument('--decoder-max-len', '-dml', type=int,
                        help='Length of longest sequence of the decoder (or bigger), 0 if you don\'t know', default=0)
    sp_test.add_argument('--create-premises','-cp', dest='create_premises', action='store_true')
    sp_test.set_defaults(create_premises=False)
    
    # # Model
    sp_test.add_argument('--model-name', type=str,
                         help='Name of the huggingface model', default='bert-base-uncased')
    sp_test.add_argument('--model-path', type=str,
                         help='Path to fined-tuned model', default=None)
    sp_test.add_argument('--model-type', type=str,
                         help='Type of the model (encode-decode or hybrid)', default='encode-decode')
    sp_test.add_argument('--checkpoints', type=str,
                         help='Checkpoint to torch model', default=None)
    sp_test.add_argument('--decoder-model-name', type=str,
                         help='Only if encoder and decoder are different', default=None)
    sp_test.add_argument('--label', '-l', type=int,
                        help='Create generative model only for one label', default=None)
    sp_test.add_argument('--hypothesis-only','-ho', dest='hypothesis_only', action='store_true')
    sp_test.add_argument('--generate-hypothesis','-gh', dest='generate_hypothesis', action='store_true')
    sp_test.set_defaults(hypothesis_only=False, generate_hypothesis=False)

    sp_comb = sp.add_parser('combine', help='Combine two models (only testing)')
    sp_comb.set_defaults(subcmd_fn=combine_models)
    sp_comb.add_argument('--data-dir-prefix', type=str,
                        help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_comb.add_argument('--modelA-path', '-map', type=str,
                         help='Path of the first model', required=True)
    sp_comb.add_argument('--modelB-path', '-mbp', type=str,
                         help='Path of the second model', required=True)
    sp_comb.add_argument('--modelA-type', '-mat', type=str,
                         help='Type of the first model', required=True)
    sp_comb.add_argument('--modelB-type', '-mbt', type=str,
                         help='Type of the second model', default=None)
    sp_comb.add_argument('--modelA-name', '-man', type=str,
                         help='Name of the first model', required=True)
    sp_comb.add_argument('--modelB-name', '-mbn', type=str,
                         help='Name of the second model', default=None)
    sp_comb.add_argument('--bs-test', '-bst', type=int,
                         help='Test batch size', default=8)
    sp_comb.add_argument('--gamma', '-g', type=float,
                         help='Gamma value', default=0.5)
    sp_comb.add_argument('--hypothesis-only','-ho', dest='hypothesis_only', action='store_true')
    sp_comb.set_defaults(hypothesis_only=False)

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed


# run_experiment("dimi", data_dir_prefix="../data/scitail/cl_scitail", bs_train=8, bs_test=4)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
