import argparse
import json
import os
import random
import sys

import numpy as np

import torch
from torch.nn import parameter
import torchvision
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AdamW, AutoModel, \
    get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, \
    get_constant_schedule_with_warmup

from src.data import PremiseGenerationDataset, DiscriminativeDataset, HypothesisOnlyDataset, DualDataset
from src.models import get_model, HybridModel
from src.train import GenerativeTrainer, DiscriminativeTrainer, OnelabelTrainer, HybridTrainer
from src.utils import FitResult, get_max_len
import math
from torch.utils.tensorboard import SummaryWriter
from torch.optim.sgd import SGD


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


def run_experiment(run_name, out_dir='./results', data_dir_prefix='./data/snli_1.0/cl_snli', model_path=None,
                   model_name='bert-base-uncased', model_type='encode-decode', decoder_model_name=None, seed=None,
                   drive=False, do_test=True,
                   # Training params
                   bs_train=16, bs_test=8, batches=0, epochs=20,
                   early_stopping=3, checkpoints=None, lr=0.001, reg=1e-3, max_len=0, decoder_max_len=0,
                   optimizer_type='Adam', momentum=0.9, word_dropout=0.0, label_smoothing_epsilon=0.0,
                   tie_embeddings=False, hypothesis_only=False, generate_hypothesis=False, rev=0.0, reduction='mean',
                   hyp_only_model=None, hard_validation=False,
                   # Model params
                   beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.0, param_freezing_ratio=0.0,
                   gradual_unfreeze=False,
                   ret_res=False, gamma=0.0,
                   # Dataset params
                   inject_bias=0, bias_ids=[30000, 30001, 30002], bias_ratio=0.5, bias_location='start', non_discriminative_bias=False,
                   label=None, threshold=0.0, attribution_map=None, move_to_hypothesis=False,
                   **kw):
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hard_test_labels = []
    hard_test_lines = []

    test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')
    val_str = ('dev_matched' if 'mnli' in data_dir_prefix else 'val')
    if hard_validation:
        val_str += '_hard'

    attribution_paths = [None,None,None,None]

    if attribution_map is not None:
        if not os.path.isdir(attribution_map):
            raise FileNotFoundError(f"There is no such folder: {attribution_map}")
        files = [f for f in os.listdir(attribution_map) if os.path.isfile(os.path.join(attribution_map, f))]
        for i,prefix in enumerate(["train_set","validation_set","test_set","hard_test_set"]):
            f = list(filter(lambda f: f.startswith(prefix), files))
            if len(f)>0:
                path_ = os.path.join(attribution_map, f[0])
                attribution_paths[i] = torch.load(path_, map_location=device)

    with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
        test_labels = test_labels_file.readlines()
    with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
    with open(data_dir_prefix + '_train_lbl_file') as train_labels_file:
        train_labels = train_labels_file.readlines()
    with open(data_dir_prefix + '_train_source_file') as train_lines_file:
        train_lines = train_lines_file.readlines()
    with open(data_dir_prefix + f'_{val_str}_lbl_file') as val_labels_file:
        val_labels = val_labels_file.readlines()
    with open(data_dir_prefix + f'_{val_str}_source_file') as val_lines_file:
        val_lines = val_lines_file.readlines()
    if os.path.isfile(data_dir_prefix + '_test_hard_lbl_file') and \
            os.path.isfile(data_dir_prefix + '_test_hard_source_file'):
        with open(data_dir_prefix + '_test_hard_lbl_file') as val_labels_file:
            hard_test_labels = val_labels_file.readlines()
        with open(data_dir_prefix + '_test_hard_source_file') as val_lines_file:
            hard_test_lines = val_lines_file.readlines()

    if 'bart' in model_name:
        model_type = 'bart'
    elif 't5' in model_name:
        model_type = 't5'

    tokenizer = AutoTokenizer.from_pretrained(model_name if 'patrick' not in model_name else 'bert-base-uncased')
    if 'gpt' in model_name:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer_decoder = None
    if decoder_model_name is not None and 'gpt' in decoder_model_name:
        from transformers import GPT2Tokenizer
        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        tokenizer_decoder = GPT2Tokenizer.from_pretrained(decoder_model_name)
        tokenizer_decoder.pad_token = tokenizer_decoder.unk_token

    size_test = batches * bs_test if batches > 0 else 10 ** 8
    size_train = batches * bs_train if batches > 0 else 10 ** 8

    all_labels_text = list(set(
        test_labels[:size_test] + train_labels[:size_train] + val_labels[:size_test] + hard_test_labels[:size_test]))
    all_labels_text.sort()
    num_labels = len(all_labels_text)

    if label is not None:
        if model_type.startswith('disc'):
            raise AttributeError("Can't specify label with discriminative model")
        # train_lines = np.array(train_lines)[np.array(train_labels)==all_labels_text[label]].tolist()
        # train_labels = np.array(train_labels)[np.array(train_labels)==all_labels_text[label]].tolist()
        # val_lines = np.array(val_lines)[np.array(val_labels)==all_labels_text[label]].tolist()
        # val_labels = np.array(val_labels)[np.array(val_labels)==all_labels_text[label]].tolist()
        # test_lines = np.array(test_lines)[np.array(test_labels)==all_labels_text[label]].tolist()
        # test_labels = np.array(test_labels)[np.array(test_labels)==all_labels_text[label]].tolist()

    elif not model_type.startswith('disc'):
        all_labels = ['[' + l.lower().strip() + ']' for l in all_labels_text]
        tokenizer.add_tokens(all_labels)
        labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]
        # labels_ids = [2870,2874,2876] if "bert" in model_name else [50000,50001,50002]
        print(f'Labels IDs: {labels_ids}')


    dataset = None
    trainer_type = None
    data_args = {"move_to_hypothesis":move_to_hypothesis}
    dataloader_args = {}
    train_args = {'reduction': reduction}

    hyp = None
    optimizer_grouped_parameters = []
    if hyp_only_model is not None:
        if os.path.isdir(hyp_only_model):
            hyp = get_model(model='discriminative',
                            model_name=decoder_model_name if decoder_model_name is not None else model_name,
                            model_path=hyp_only_model, num_labels=num_labels)
            hyp.requires_grad_ = False
        else:
            hyp = get_model(model='discriminative',
                            model_name=decoder_model_name if decoder_model_name is not None else model_name,
                            model_path=None, num_labels=num_labels)

            optimizer_grouped_parameters = [
                {
                    "params": hyp.parameters()
                }
            ]

        hyp = hyp.to(device)
        train_args['hyp_prior_model'] = hyp

    if model_type in ['encode-decode', 'bart', 'shared', 't5', 'decoder-only']:
        dataset = DiscriminativeDataset
        if label is None:
            trainer_type = GenerativeTrainer
        else:
            trainer_type = OnelabelTrainer
            train_args['label'] = label

        train_args['rev'] = rev
        train_args['possible_labels_ids'] = labels_ids
        train_args['epsilon'] = label_smoothing_epsilon
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['gradual_unfreeze'] = gradual_unfreeze
        train_args['gamma'] = gamma
        # dataloader_args['collate_fn'] = my_collate
    elif model_type.startswith('disc'):
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

    if model_type == 'decoder-only':
        train_args['decoder_only'] = True

    # import pdb; pdb.set_trace()
    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[2]
    ds_test = dataset(test_lines, test_labels, tokenizer, max_len=max_len, **data_args)

    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[1]
    ds_val = dataset(val_lines, val_labels, tokenizer, max_len=max_len, **data_args)
    # data_args['dropout'] = word_dropout
    data_args.update({
        'inject_bias': inject_bias,
        'bias_ids': bias_ids,
        'bias_ratio': bias_ratio,
        'bias_location': bias_location,
        'non_discriminative_bias': non_discriminative_bias,
        'dropout': word_dropout,
        'threshold': threshold,
    })
    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[0]
    ds_train = dataset(train_lines, train_labels, tokenizer, max_len=max_len, **data_args)
    # import pdb; pdb.set_trace()
    if batches > 0:
        # ds_test = Subset(ds_test, range(batches * bs_test))
        # ds_val = Subset(ds_val, range(batches * bs_test))
        ds_train = Subset(ds_train, range(batches * bs_train))

    model = get_model(tokenizer=tokenizer,
                      tokenizer_decoder=tokenizer_decoder,
                      model=model_type,
                      model_name=model_name,
                      decoder_model_name=decoder_model_name,
                      model_path=model_path,
                      param_freezing_ratio=param_freezing_ratio,
                      tie_embeddings=tie_embeddings,
                      label=label,
                      gamma=gamma)

    # import pdb; pdb.set_trace()
    if torch.cuda.device_count()>1 and hasattr(model, 'parallelize'):
        n_devices = torch.cuda.device_count()
        num_layers = model.config.num_layers if hasattr(model.config,'num_layers') else model.config.n_layer
        k = num_layers//n_devices
        device_map = {i:list(range(i*k,(i+1)*k)) for i in range(n_devices)}
        # device_map = {0:[0], 1:list(range(1,num_layers))}
        model.parallelize(device_map)
    else:
        model.to(device)

    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=True, **dataloader_args)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_test, shuffle=False, **dataloader_args)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False, **dataloader_args)

    optimizer_grouped_parameters.append(
        {
            'params':model.parameters()
        }
    )

    if optimizer_type.lower() == 'adam':
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise AttributeError('only SGD and Adam supported for now')

    # import pdb; pdb.set_trace()
    num_batches = batches if batches > 0 else len(dl_train)
    num_steps = epochs * num_batches
    print(f"Number of training steps: {num_steps}")
    scheduler = None
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2*num_batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2 * num_batches,
                                                num_training_steps=num_steps)
    writer = SummaryWriter()
    trainer = trainer_type(model, optimizer, scheduler, max_len=max_len, device=device, **train_args)
    fit_res = trainer.fit(dl_train, dl_val, num_epochs=epochs, early_stopping=early_stopping, checkpoints=checkpoints,
                          drive=drive, writer=writer)

    if ret_res:
        del model
        return fit_res.test_acc[-4] if len(fit_res.test_acc) >= 4 else fit_res.test_acc[-1]
    save_experiment(run_name, out_dir, cfg, fit_res)

    if do_test:
        trainer.test(dl_test, writer=writer)
        if len(hard_test_labels) > 0 and len(hard_test_lines) > 0:
            if attribution_map is not None:
                data_args['attribution_map'] = attribution_paths[3]
            else:
                data_args.pop('attribution_map',None)
            ds_hard_test = dataset(hard_test_lines, hard_test_labels, tokenizer, max_len=max_len, **data_args)
            if batches > 0:
                ds_test = Subset(ds_hard_test, range(batches * bs_test))
            dl_hard_test = torch.utils.data.DataLoader(ds_hard_test, bs_test, shuffle=False)
            trainer.test(dl_hard_test, writer=writer)


def test_model(run_name, out_dir='./results_test', data_dir_prefix='./data/snli_1.0/cl_snli',
               model_name='bert-base-uncased', model_path=None, model_type='encode-decode',
               decoder_model_name=None, seed=None, save_results=None,
               # Training params
               bs_test=8, batches=0,
               checkpoints=None, max_len=0, decoder_max_len=0,
               hypothesis_only=False, generate_hypothesis=False, create_premises=False,
               label=0, attribution_map=None, move_to_hypothesis=False, hyp_only_model=None,
               **kw):
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = 1
    cfg = locals()

    tf = torchvision.transforms.ToTensor()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attribution_paths = [None,None]

    if attribution_map is not None:
        if not os.path.isdir(attribution_map):
            raise FileNotFoundError(f"There is no such folder: {attribution_map}")
        files = [f for f in os.listdir(attribution_map) if os.path.isfile(os.path.join(attribution_map, f))]
        for i,prefix in enumerate(["test_set","hard_test_set"]):
            f = list(filter(lambda f: f.startswith(prefix), files))
            if len(f)>0:
                path_ = os.path.join(attribution_map, f[0])
                attribution_paths[i] = torch.load(path_, map_location=device)

    hard_test_labels = None
    hard_test_lines = None

    test_str = ('test_matched_unlabeled' if 'mnli' in data_dir_prefix else 'test')
    val_str = ('dev_matched' if 'mnli' in data_dir_prefix else 'val')

    with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
        test_labels = test_labels_file.readlines()
    with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
    with open(data_dir_prefix + f'_{val_str}_lbl_file') as val_labels_file:
        val_labels = val_labels_file.readlines()
    with open(data_dir_prefix + f'_{val_str}_source_file') as val_lines_file:
        val_lines = val_lines_file.readlines()
    if os.path.isfile(data_dir_prefix + f'_{test_str}_hard_lbl_file') and \
            os.path.isfile(data_dir_prefix + f'_{test_str}_hard_source_file'):
        with open(data_dir_prefix + f'_{test_str}_hard_lbl_file') as val_labels_file:
            hard_test_labels = val_labels_file.readlines()
        with open(data_dir_prefix + f'_{test_str}_hard_source_file') as val_lines_file:
            hard_test_lines = val_lines_file.readlines()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'gpt' in model_name:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer_decoder = None
    if decoder_model_name is not None and 'gpt' in decoder_model_name:
        from transformers import GPT2Tokenizer
        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        tokenizer_decoder = GPT2Tokenizer.from_pretrained(decoder_model_name)
        tokenizer_decoder.pad_token = tokenizer_decoder.unk_token

    size_test = batches * bs_test if batches > 0 else 10 ** 8

    all_labels_text = list(set(val_labels[:size_test]))
    all_labels_text.sort()
    num_labels = len(all_labels_text)

    if not model_type.startswith('disc') and label is None:
        all_labels = ['[' + l.lower().replace('\n', '') + ']' for l in all_labels_text]
        tokenizer.add_tokens(all_labels)
        labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]
        # labels_ids = [2870,2874,2876]
        print(f'Labels IDs: {labels_ids}')
        # if tokenizer_decoder is not None:
        #     tokenizer_decoder.add_tokens(all_labels)
        #     labels_ids_decoder = [tokenizer_decoder.encode(label, add_special_tokens=False)[0] for label in all_labels]
        #     print(f'Labels IDs for decoder: {labels_ids_decoder}')

    dataset = None
    trainer_type = None
    data_args = {"move_to_hypothesis":move_to_hypothesis}
    dataloader_args = {}
    train_args = {}

    if hyp_only_model is not None:
        if os.path.isdir(hyp_only_model):
            hyp = get_model(model='discriminative',
                            model_name=decoder_model_name if decoder_model_name is not None else model_name,
                            model_path=hyp_only_model, num_labels=num_labels)
            hyp.requires_grad_ = False
        else:
            raise AssertionError("'hyp_only_model' must be a path for a pre-trained model when testing")

        hyp = hyp.to(device)
        train_args['hyp_prior_model'] = hyp

    train_args['save_results'] = save_results
    if model_type in ['encode-decode', 'bart', 'shared']:
        dataset = DiscriminativeDataset
        if label is None:
            trainer_type = GenerativeTrainer
        else:
            trainer_type = OnelabelTrainer
        # data_args['tokenizer_decoder'] = tokenizer_decoder
        # data_args['generate_hypothesis'] = generate_hypothesis
        train_args['possible_labels_ids'] = labels_ids
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['create_premises'] = create_premises
        # dataloader_args['collate_fn'] = my_collate
    elif model_type.startswith('disc'):
        if hypothesis_only:
            dataset = HypothesisOnlyDataset
        else:
            dataset = DiscriminativeDataset
        trainer_type = DiscriminativeTrainer
        train_args['num_labels'] = num_labels
        train_args['tokenizer'] = tokenizer

    # import pdb; pdb.set_trace()
    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[0]
    if 'mnli' in data_dir_prefix:
        train_args['mnli_ids_path'] = 'other/mnli_ids.csv'
    ds_test = dataset(test_lines, test_labels, tokenizer, max_len=max_len, **data_args)

    data_args.pop('mnli_ids_path',None)
    ds_val = dataset(val_lines, val_labels, tokenizer, max_len=max_len, **data_args)

    if batches > 0:
        ds_test = Subset(ds_test, range(batches * bs_test))
        ds_val = Subset(ds_val, range(batches * bs_test))

    model = get_model(tokenizer=tokenizer, tokenizer_decoder=tokenizer_decoder, model=model_type, model_name=model_name,
                      decoder_model_name=decoder_model_name, model_path=model_path)

    model.to(device)

    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False, **dataloader_args)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_test, shuffle=False, **dataloader_args)

    writer = None
    if checkpoints is None:
        writer = SummaryWriter()

    trainer = trainer_type(model, optimizer=None, scheduler=None, device=device, **train_args)
    fit_res = trainer.test(dl_test, checkpoints=checkpoints, writer=writer)
    save_experiment(run_name, out_dir, cfg, fit_res)
    if hard_test_labels is not None and hard_test_lines is not None:
        if hasattr(trainer, 'save_results') and trainer.save_results is not None:
            trainer.save_results += '_hard'
        if attribution_map is not None:
            data_args['attribution_map'] = attribution_paths[1]
        if 'mnli' in data_dir_prefix and save_results is not None:
            trainer._get_ids_for_mnli('other/mnli_hard_ids.csv')
            trainer.index=0
        
        ds_hard_test = dataset(hard_test_lines, hard_test_labels, tokenizer, max_len=max_len, **data_args)
        if batches > 0:
            ds_test = Subset(ds_hard_test, range(batches * bs_test))
        dl_hard_test = torch.utils.data.DataLoader(ds_hard_test, bs_test, shuffle=False, **dataloader_args)
        fit_res = trainer.test(dl_hard_test, checkpoints=checkpoints, writer=writer)
        save_experiment(run_name + '_hard', out_dir, cfg, fit_res)


def combine_models(data_dir_prefix='./data/snli_1.0/cl_snli', bs_test=8,
                   modelA_path=None, modelA_type=None,
                   modelA_name=None, modelB_path=None,
                   modelB_type=None, modelB_name=None,
                   gamma=0.5, hypothesis_only=False):
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
    model = HybridModel(modelA, modelB, gamma)

    model.to(device)

    tokenizerA = AutoTokenizer.from_pretrained(modelA_name)

    size_test = 10 ** 8

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
    DatasetA = datasetA(test_lines, test_labels, tokenizerA)

    datasetB = HypothesisOnlyDataset if hypothesis_only else DiscriminativeDataset
    DatasetB = datasetB(test_lines, test_labels, tokenizerB)

    dataset = DualDataset(DatasetA, DatasetB)
    dataloader = torch.utils.data.DataLoader(dataset, bs_test, shuffle=False)

    trainer = DualTesterTrainer(model, tokenizer=tokenizerB, optimizer=None, scheduler=None, max_len=128, num_labels=3,
                                possible_labels_ids=labels_ids, device=device)
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
    sp_exp = sp.add_parser('train', help='Train a model')
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
    # inject_bias=0, bias_ids=30000, bias_ratio=0.5, bias_location='start', seed=42
    sp_exp.add_argument('--inject-bias', type=int,
                        help='Select number of labels to inject bias to their corresponding hypotheses',
                        default=0)
    sp_exp.add_argument('--bias-ids', type=int, nargs='+', help='Select the ids of the biases symbols',
                        default=[30000, 30001, 30002])
    sp_exp.add_argument('--bias-ratio', type=float,
                        help='Select the percentege of labels to inject bias to their corresponding hypotheses',
                        default=0.5)
    sp_exp.add_argument('--bias-location', type=str,
                        help='Select where in the hypotheses to inject the bias, can be either "start" or "end", otherwise will be random location',
                        default='start')
    sp_exp.add_argument('--non-discriminative-bias', '-ndb', dest='non_discriminative_bias', action='store_true')
    sp_exp.add_argument('--attribution-map', '-am', type=str, 
                        help='path of attribution maps folder',
                        default=None)
    sp_exp.add_argument('--move-to-hypothesis', '-mth', dest='move_to_hypothesis', action='store_true')

    # # Training
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=16, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-test', type=int, help='Test batch size',
                        default=8, metavar='BATCH_SIZE')
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
    sp_exp.add_argument('--reduction', '-reduce', type=str,
                        help='How to reduce loss, can be "sum" or "mean"', default="mean")
    sp_exp.add_argument('--momentum', '-m', type=float,
                        help='Momentum for SGD', default=0.9)
    sp_exp.add_argument('--word-dropout', '-wdo', type=float,
                        help='Word dropout rate during training', default=0.0)
    sp_exp.add_argument('--label-smoothing-epsilon', '-lse', type=float,
                        help='Epsilon argument for label smoothing (does not uses labels smoothing by \
                        default', default=0.0)
    sp_exp.add_argument('--hyp-only-model', '-hom', type=str,
                        help='If you want to weigh loss by htpothesis only output', default=None)
    sp_exp.add_argument('--threshold', '-th', type=float, default=0.0)
    
    sp_exp.add_argument('--tie-embeddings', '-te', dest='tie_embeddings', action='store_true')
    sp_exp.add_argument('--hypothesis-only', '-ho', dest='hypothesis_only', action='store_true')
    sp_exp.add_argument('--gradual-unfreeze', '-gu', dest='gradual_unfreeze', action='store_true')
    sp_exp.add_argument('--generate-hypothesis', '-gh', dest='generate_hypothesis', action='store_true')
    sp_exp.add_argument('--hard-validation', '-hv', dest='hard_validation', action='store_true')
    sp_exp.add_argument('--label', '-l', type=int,
                        help='Create generative model only for one label', default=None)
    sp_exp.add_argument('--rev', type=float,
                        help='For hinge loss', default=0.0)
    sp_exp.set_defaults(tie_embeddings=False, hypothesis_only=False,
                        generate_hypothesis=False, non_discriminative_bias=False, gradual_unfreeze=False,
                        hard_validation=False)

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
    sp_exp.add_argument('--gamma', type=float, default=0.0)
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
                         default=8, metavar='BATCH_SIZE')
    sp_test.add_argument('--batches', type=int,
                         help='Number of batches per epoch, pass "0" if you want the full database', default=0)
    sp_test.add_argument('--data-dir-prefix', type=str,
                         help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_test.add_argument('--max-len', type=int,
                         help='Length of longest sequence (or bigger), 0 if you don\'t know', default=0)
    sp_test.add_argument('--decoder-max-len', '-dml', type=int,
                         help='Length of longest sequence of the decoder (or bigger), 0 if you don\'t know', default=0)
    sp_test.add_argument('--create-premises', '-cp', dest='create_premises', action='store_true')

    sp_test.add_argument('--attribution-map', '-am', type=str, 
                        help='path of attribution maps folder',
                        default=None)
    sp_test.add_argument('--move-to-hypothesis', '-mth', dest='move_to_hypothesis', action='store_true')
    sp_test.add_argument('--hyp-only-model', '-hom', type=str,
                        help='If you want to weigh loss by htpothesis only output', default=None)
    sp_test.set_defaults(create_premises=False, move_to_hypothesis=False)


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
    sp_test.add_argument('--hypothesis-only', '-ho', dest='hypothesis_only', action='store_true')
    sp_test.add_argument('--generate-hypothesis', '-gh', dest='generate_hypothesis', action='store_true')
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
    sp_comb.add_argument('--hypothesis-only', '-ho', dest='hypothesis_only', action='store_true')
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
