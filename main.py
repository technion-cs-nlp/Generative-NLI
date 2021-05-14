import argparse
import json
import os
import pdb
import random
import sys
from threading import Thread

import numpy as np

import torch
from torch.nn import parameter
from torch.utils import data
import torchvision
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AdamW, AutoModel, \
    get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, \
    get_constant_schedule_with_warmup

from src.data import PremiseGenerationDataset, DiscriminativeDataset, HypothesisOnlyDataset, DualDataset
from src.models import get_model, HybridModel
from src.train import GenerativeTrainer, DiscriminativeTrainer, OnelabelTrainer
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
                   drive=False, do_test=True, gen_premise='', 
                   # Training params
                   bs_train=16, bs_test=8, batches=0, epochs=20,
                   early_stopping=3, checkpoints=None, lr=1e-5, reg=1e-3, max_len=0, decoder_max_len=0,
                   optimizer_type='Adam', momentum=0.9, word_dropout=0.0, label_smoothing_epsilon=0.0,
                   tie_embeddings=False, hypothesis_only=False, generate_hypothesis=False, reverse=False, reduction='sum',
                   hyp_only_model=None, hard_validation=False, merge_train=False, test_with_prior=False, sched='linear',
                   # Model params
                   beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.0, param_freezing_ratio=0.0,
                   gradual_unfreeze=False,
                   ret_res=False, gamma=0.0, tie_encoder_decoder=False, 
                   # Dataset params
                   inject_bias=0, bias_ids=[30000, 30001, 30002], bias_ratio=0.5, bias_location='start', non_discriminative_bias=False,
                   label=None, threshold=0.0, attribution_map=None, move_to_hypothesis=False, filt_method='true', train_hyp=False, 
                   attribution_tokenizer=None, premise_only=False, cheat=False, calc_uniform=False, pure_gen=False):
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        for i,prefix in enumerate(["train_set","val_set","test_set","hard_test_set"]):
            f = list(filter(lambda f: f.startswith(prefix), files))
            if len(f)>0:
                path_ = os.path.join(attribution_map, f[0])
                attribution_paths[i] = torch.load(path_, map_location=torch.device('cpu'))

    if gen_premise != '':
        gen_premise += '_'

    if data_dir_prefix == 'fever':
        from src.data_miki.datasets import load_dataset_aux
        fever, _ = load_dataset_aux('fever_nli')
        train_lines = fever['train']
        # train_lines = train_lines.filter(lambda x: x['label']!=2)
        train_labels = None
        val_lines = fever['validation']
        val_labels = None
        fever_test, _ = load_dataset_aux('fever_symmetric')
        test_lines = fever_test['test']
        test_labels = None
    else:
        with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
            test_labels = test_labels_file.readlines()
        
        with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
            test_lines = test_lines_file.readlines()
        
        with open(data_dir_prefix + f'_train_{gen_premise}lbl_file') as train_labels_file:
            train_labels = train_labels_file.readlines()
        
        with open(data_dir_prefix + f'_train_{gen_premise}source_file') as train_lines_file:
            train_lines = train_lines_file.readlines()
        
        with open(data_dir_prefix + f'_{val_str}_lbl_file') as val_labels_file:
            val_labels = val_labels_file.readlines()
        
        with open(data_dir_prefix + f'_{val_str}_source_file') as val_lines_file:
            val_lines = val_lines_file.readlines()
        
        if os.path.isfile(data_dir_prefix + f'_{test_str}_hard_lbl_file') and \
                os.path.isfile(data_dir_prefix + f'_{test_str}_hard_source_file'):
            with open(data_dir_prefix + f'_{test_str}_hard_lbl_file') as hard_test_labels_file:
                hard_test_labels = hard_test_labels_file.readlines()
                
            with open(data_dir_prefix + f'_{test_str}_hard_source_file') as hard_test_lines_file:
                hard_test_lines = hard_test_lines_file.readlines()

    # import pdb; pdb.set_trace()

    if gen_premise != "":
        dict_count = {}
        for line in train_lines:
            if line not in dict_count:
                    dict_count[line]=0
            dict_count[line]+=1
        similar = [line for line,count in dict_count.items() if count==3]
        mask = ~np.isin(np.array(train_lines),similar)
        train_labels = np.array(train_labels)[mask].tolist()
        train_lines = np.array(train_lines)[mask].tolist()
        # import pdb; pdb.set_trace()
        pass

    if merge_train and gen_premise != '':
        with open(data_dir_prefix + f'_train_lbl_file') as train_labels_file:
            train_labels += train_labels_file.readlines()
        with open(data_dir_prefix + f'_train_source_file') as train_lines_file:
            train_lines += train_lines_file.readlines()

    
    # if 'bart' in model_name:
    #     model_type = 'bart'
    # elif 't5' in model_name:
    #     model_type = 't5'

    tokenizer = AutoTokenizer.from_pretrained(model_name if 'patrick' not in model_name else 'bert-base-uncased')
    if 'gpt' in model_name:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer_decoder = None
    if decoder_model_name is not None and 'gpt' in decoder_model_name:
        from transformers import GPT2Tokenizer
        GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
        tokenizer_decoder = GPT2Tokenizer.from_pretrained(decoder_model_name)
        tokenizer_decoder.pad_token = tokenizer_decoder.unk_token

    # size_test = batches * bs_test if batches > 0 else 10 ** 8
    size_test = 10**8
    size_train = batches * bs_train if batches > 0 else 10 ** 8

    if data_dir_prefix == 'fever':
        all_labels_text = ["SUPPORTS", "REFUTES", "NOT-ENOUGH-INFO"]
        # all_labels_text = ['A','B','C']
        # all_labels_text = ['A','B']
    else:
        all_labels_text = list(set(
            # test_labels[:size_test] + train_labels[:size_train] + val_labels[:size_test] + hard_test_labels[:size_test]))
            test_labels + train_labels + val_labels + hard_test_labels))
    all_labels_text.sort()
    ratios = None
    # if calc_uniform:
    #     ratios = [1/3,1/3,1/3]
    #     # pdb.set_trace()
    #     for i,l in enumerate(all_labels_text):
    #         rate = len([sam for sam in train_labels if sam==l]) / len(train_labels)
    #         ratios[i] = rate
    
    num_labels = len(all_labels_text)
    # import pdb; pdb.set_trace()
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
    data_args = {"move_to_hypothesis":move_to_hypothesis, 'possible_labels':all_labels_text, 'rev':reverse, 'pure_gen':pure_gen}
    dataloader_args = {}
    train_args = {'reduction': reduction, 'ratios':ratios}

    hyp = None
    optimizer_grouped_parameters = []
    if hyp_only_model is not None:
        if torch.cuda.device_count() > 1:
            hyp_device = torch.device('cuda:1')
        else:
            hyp_device = device
        if os.path.isdir(hyp_only_model):
            hyp = get_model(model='discriminative',
                            model_name=decoder_model_name if decoder_model_name is not None else model_name,
                            model_path=hyp_only_model, num_labels=num_labels)
            hyp = hyp.to(hyp_device)
            if train_hyp:
                optimizer_grouped_parameters = [
                    {
                        "params": hyp.parameters()
                    }
                ]
                hyp.requires_grad_ = True
            else: 
                hyp.requires_grad_ = False
        else:
            hyp = get_model(model='discriminative',
                            model_name=decoder_model_name if decoder_model_name is not None else model_name,
                            model_path=None, num_labels=num_labels)
            hyp = hyp.to(hyp_device)
            optimizer_grouped_parameters = [
                {
                    "params": hyp.parameters()
                }
            ]

        train_args['hyp_prior_model'] = hyp
        train_args['test_with_prior'] = test_with_prior

    dataset = DiscriminativeDataset

    if model_type in ['encode-decode', 'bart', 'shared', 't5', 'decoder-only', 'bert2bert']:
        if label is None:
            trainer_type = GenerativeTrainer
        else:
            trainer_type = OnelabelTrainer
            train_args['label'] = label

        # train_args['rev'] = rev
        train_args['possible_labels_ids'] = labels_ids
        train_args['epsilon'] = label_smoothing_epsilon
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['gradual_unfreeze'] = gradual_unfreeze
        train_args['gamma'] = gamma
        # dataloader_args['collate_fn'] = my_collate
    elif model_type.startswith('disc'):
        trainer_type = DiscriminativeTrainer
        train_args['num_labels'] = num_labels
        train_args['tokenizer'] = tokenizer

    elif model_type == 'hybrid':
        trainer_type = HybridTrainer
        train_args['possible_labels_ids'] = labels_ids
        train_args['epsilon'] = label_smoothing_epsilon
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['gradual_unfreeze'] = gradual_unfreeze
        train_args['num_labels'] = num_labels

    if model_type == 'decoder-only':
        train_args['decoder_only'] = True
    if premise_only:
        data_args['premise_only'] = premise_only

    data_args['threshold'] = threshold
    data_args['filt_method'] = filt_method
    if attribution_tokenizer is None:
        attribution_tokenizer = model_name
    data_args['attribution_tokenizer'] = attribution_tokenizer

    data_dict = {
        'inject_bias': inject_bias,
        'bias_ids': bias_ids,
        'bias_ratio': bias_ratio,
        'bias_location': bias_location,
        'non_discriminative_bias': non_discriminative_bias,
        'hypothesis_only':hypothesis_only,
    }
    data_args.update(data_dict)
    # import pdb; pdb.set_trace()
    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[2]
    ds_test = dataset(test_lines, test_labels, tokenizer, max_len=max_len, **data_args)

    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[1]
    ds_val = dataset(val_lines, val_labels, tokenizer, max_len=max_len, **data_args)
    # data_args['dropout'] = word_dropout

    if cheat:        # cheat
        if attribution_map is not None:
            data_args['attribution_map'] = attribution_paths[3]
        else:
            data_args.pop('attribution_map',None)
        ds_val = dataset(hard_test_lines, hard_test_labels, tokenizer, max_len=max_len, **data_args)
        # if batches > 0:
        #     ds_val = Subset(ds_hard_test, range(batches * bs_test))
        # dl_hard_test = torch.utils.data.DataLoader(ds_hard_test, bs_test, shuffle=False, **dataloader_args)
        # dl_val = dl_hard_test

    train_dict = {
        # 'inject_bias': inject_bias,
        # 'bias_ids': bias_ids,
        # 'bias_ratio': bias_ratio,
        # 'bias_location': bias_location,
        # 'non_discriminative_bias': non_discriminative_bias,
        'dropout': word_dropout,
    }
    data_args.update(train_dict)
    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[0]
    data_args['filt_method'] = filt_method if filt_method != 'none' else 'true'
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
                      gamma=gamma,
                      tie_encoder_decoder=tie_encoder_decoder,
                      num_labels=num_labels)

    # model.config.min_length = 5
    # model.config.max_length = 64
    # model.config.task_specific_params['summarization']['min_length'] = 5
    # model.config.task_specific_params['summarization']['max_length'] = 64

    # import pdb; pdb.set_trace()
    if torch.cuda.device_count()>1 and hyp is None:
        if hasattr(model, 'parallelize'):
            n_devices = torch.cuda.device_count()
            num_layers = model.config.num_layers if hasattr(model.config,'num_layers') else model.config.n_layer
            k = num_layers//n_devices
            device_map = {i:list(range(i*k,(i+1)*k)) for i in range(n_devices)}
            # device_map = {0:[0], 1:list(range(1,num_layers))}
            model.parallelize(device_map)
        elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            pass
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
    if 'linear' in sched.lower():
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_batches,
                                                num_training_steps=num_steps)
    elif 'warmup' in sched.lower():
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_batches)
    elif 'cosine' in sched.lower():
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_batches,
                                                num_training_steps=num_steps)
    writer = SummaryWriter()
    trainer = trainer_type(model, optimizer, scheduler, max_len=max_len, device=device, **train_args)
    fit_res = trainer.fit(dl_train, dl_val, num_epochs=epochs, early_stopping=early_stopping, checkpoints=checkpoints,
                          drive=drive, writer=writer)
    
    if ret_res:
        del model
        return fit_res.test_acc[-4] if len(fit_res.test_acc) >= 4 else fit_res.test_acc[-1]
    save_experiment(run_name, out_dir, cfg, fit_res)

    del model
    if hyp is not None:
        del hyp
    torch.cuda.empty_cache()
    
    if do_test:
        print('_'*50)
        test_model(run_name, out_dir+"_test", data_dir_prefix, model_name, checkpoints+"_model", model_type, decoder_model_name, seed, None,
                    bs_test, batches, None, 0, 0, hypothesis_only, False, False, label, attribution_map,
                    move_to_hypothesis, hyp_only_model, threshold, reduction, filt_method, attribution_tokenizer=attribution_tokenizer, test_with_prior=test_with_prior,
                    premise_only=premise_only, calc_uniform=calc_uniform, reverse=reverse, pure_gen=pure_gen,
                    inject_bias=inject_bias, bias_ids=bias_ids, bias_ratio=bias_ratio, bias_location=bias_location, non_discriminative_bias=non_discriminative_bias,)



def test_model(run_name, out_dir='./results_test', data_dir_prefix='./data/snli_1.0/cl_snli',
               model_name='bert-base-uncased', model_path=None, model_type='encode-decode',
               decoder_model_name=None, seed=None, save_results=None,
               # Training params
               bs_test=8, batches=0,
               checkpoints=None, max_len=0, decoder_max_len=0,
               hypothesis_only=False, generate_hypothesis=False, create_premises=False,
               label=None, attribution_map=None, move_to_hypothesis=False, hyp_only_model=None, threshold=0.0,
               reduction='sum', filt_method='true', attribution_tokenizer=None, test_with_prior=False,
               premise_only=False, calc_uniform=False, reverse=False, pure_gen=False,
               inject_bias=0, bias_ids=[30000, 30001, 30002], bias_ratio=0.5, bias_location='start', non_discriminative_bias=False,
               save_likelihoods=None, val=False):
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
                attribution_paths[i] = torch.load(path_, map_location=torch.device('cpu'))

    hard_test_labels = None
    hard_test_lines = None

    if save_results is not None:
        test_str = ('test_matched_unlabeled' if 'mnli' in data_dir_prefix else 'test')
    else:
        if 'hans' in data_dir_prefix:
            test_str = ''
        else:
            test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')
    val_str = ('dev_matched' if 'mnli' in data_dir_prefix else 'val')
    hard_val_str = val_str + '_hard'

    train_labels = []
    train_lines = []
    val_labels = []
    val_lines = []
    hard_val_labels = []
    hard_val_lines = []

    if data_dir_prefix == 'fever':
        from src.data_miki.datasets import load_dataset_aux
        fever, _ = load_dataset_aux('fever_nli')
        # train_lines = fever['train']
        # train_labels = None
        # test_lines = fever['validation']
        # test_labels = None
        fever_test, _ = load_dataset_aux('fever_symmetric')
        test_lines = fever_test['test']
        test_labels = None
        fever_testv2, _ = load_dataset_aux('fever_symmetricv2')
        hard_test_lines = fever_testv2['test']
        hard_test_labels = None
    else:
        # pdb.set_trace()
        with open(data_dir_prefix + f'_{test_str}{"_" if test_str!="" else ""}lbl_file') as test_labels_file:
            test_labels = test_labels_file.readlines()
        with open(data_dir_prefix + f'_{test_str}{"_" if test_str!="" else ""}source_file') as test_lines_file:
            test_lines = test_lines_file.readlines()

        if os.path.isfile(data_dir_prefix + f'_train_lbl_file'):
            with open(data_dir_prefix + f'_train_lbl_file') as train_labels_file:
                train_labels = train_labels_file.readlines()
            with open(data_dir_prefix + f'_train_source_file') as train_lines_file:
                train_lines = train_lines_file.readlines()

        if os.path.isfile(data_dir_prefix + f'_{val_str}_lbl_file'):
            with open(data_dir_prefix + f'_{val_str}_lbl_file') as val_labels_file:
                val_labels = val_labels_file.readlines()
            with open(data_dir_prefix + f'_{val_str}_source_file') as val_lines_file:
                val_lines = val_lines_file.readlines()

        if os.path.isfile(data_dir_prefix + f'_{hard_val_str}_lbl_file'):
            with open(data_dir_prefix + f'_{hard_val_str}_lbl_file') as hard_val_labels_file:
                hard_val_labels = hard_val_labels_file.readlines()
            with open(data_dir_prefix + f'_{hard_val_str}_source_file') as hard_val_lines_file:
                    hard_val_lines = hard_val_lines_file.readlines()

        if os.path.isfile(data_dir_prefix + f'_{test_str}_hard_lbl_file') and \
                os.path.isfile(data_dir_prefix + f'_{test_str}_hard_source_file'):
            with open(data_dir_prefix + f'_{test_str}_hard_lbl_file') as hard_test_labels_file:
                hard_test_labels = hard_test_labels_file.readlines()
            with open(data_dir_prefix + f'_{test_str}_hard_source_file') as hard_test_lines_file:
                hard_test_lines = hard_test_lines_file.readlines()         

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

    if data_dir_prefix == 'fever':
        all_labels_text = ["SUPPORTS", "REFUTES", "NOT-ENOUGH-INFO"]
        # all_labels_text = ['A','B','C']
        # all_labels_text = ['A','B']
    elif 'hans' in data_dir_prefix:
        all_labels_text = ["contradiction\n", "entailment\n", "neutral\n"]
    else:
        size_train = 10**8
        all_labels_text = list(set(
            # test_labels[:size_test] + train_labels[:size_train] + val_labels[:size_test] + hard_test_labels[:size_test]))
            train_labels# + test_labels + val_labels# + (hard_test_labels if hard_test_labels is not None else [])
            ))
    all_labels_text.sort()  
    ratios = None
    # if calc_uniform:
    #     ratios = [1/3,1/3,1/3]
    #     # pdb.set_trace()
    #     for i,l in enumerate(all_labels_text):
    #         rate = len([sam for sam in train_labels if sam==l]) / len(train_labels)
    #         ratios[i] = rate

    num_labels = len(all_labels_text)

    if not model_type.startswith('disc') and label is None:
        all_labels = ['[' + l.lower().strip() + ']' for l in all_labels_text]
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
    # pdb.set_trace()
    data_args = {"move_to_hypothesis":move_to_hypothesis, 'possible_labels':all_labels_text, 'rev':reverse, 'pure_gen':pure_gen}
    dataloader_args = {}
    train_args = {'reduction':reduction, 'ratios':ratios, 'save_likelihoods':save_likelihoods}
    if 'hans' in data_dir_prefix:
        train_args['hans']=True
    hyp = None
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
        train_args['test_with_prior'] = test_with_prior
    if ratios is not None:
        train_args['test_with_prior'] = test_with_prior

    train_args['save_results'] = save_results
    if model_type in ['encode-decode', 'bart', 'shared', 'decoder-only', 'bert2bert']:
        dataset = DiscriminativeDataset
        if label is None:
            trainer_type = GenerativeTrainer
        else:
            trainer_type = OnelabelTrainer
        # data_args['tokenizer_decoder'] = tokenizer_decoder
        # data_args['generate_hypothesis'] = generate_hypothesis
        # pdb.set_trace()
        train_args['possible_labels_ids'] = labels_ids
        train_args['tokenizer_encoder'] = tokenizer
        train_args['tokenizer_decoder'] = tokenizer_decoder
        train_args['create_premises'] = create_premises
        # dataloader_args['collate_fn'] = my_collate
    elif model_type.startswith('disc'):
        if hypothesis_only or premise_only:
            dataset = DiscriminativeDataset
            data_args['premise_only'] = premise_only
        else:
            dataset = DiscriminativeDataset
        trainer_type = DiscriminativeTrainer
        train_args['num_labels'] = num_labels
        train_args['tokenizer'] = tokenizer

    if model_type == 'decoder-only':
        train_args['decoder_only'] = True

    data_args['threshold'] = threshold
    data_args['filt_method'] = filt_method
    if attribution_tokenizer is None:
        attribution_tokenizer = model_name
    data_args['attribution_tokenizer'] = attribution_tokenizer

    # import pdb; pdb.set_trace()
    if attribution_map is not None:
        data_args['attribution_map'] = attribution_paths[0]
    if 'mnli' in data_dir_prefix:
        train_args['mnli_ids_path'] = 'other/mnli_ids.csv'

    data_dict = {
        'inject_bias': inject_bias,
        'bias_ids': bias_ids,
        'bias_ratio': bias_ratio,
        'bias_location': bias_location,
        'non_discriminative_bias': non_discriminative_bias,
        'hypothesis_only': hypothesis_only,
    }
    data_args.update(data_dict)

    ds_test = dataset(test_lines, test_labels, tokenizer, **data_args)

    data_args.pop('mnli_ids_path',None)
    data_args.pop('attribution_map',None)
    ds_val = dataset(val_lines, val_labels, tokenizer, **data_args)
    ds_hard_val = dataset(hard_val_lines, hard_val_labels, tokenizer, **data_args)
    
    ds_train = dataset(train_lines, train_labels, tokenizer, **data_args)

    if batches > 0:
        ds_test = Subset(ds_test, range(batches * bs_test))
        # ds_val = Subset(ds_val, range(batches * bs_test))

    model = get_model(tokenizer=tokenizer, tokenizer_decoder=tokenizer_decoder, model=model_type, model_name=model_name,
                      decoder_model_name=decoder_model_name, model_path=model_path, num_labels=num_labels)

    model.to(device)

    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=False, **dataloader_args)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_test, shuffle=False, **dataloader_args)
    dl_hard_val = torch.utils.data.DataLoader(ds_hard_val, bs_test, shuffle=False, **dataloader_args)
    dl_train = torch.utils.data.DataLoader(ds_train, bs_test, shuffle=False, **dataloader_args)

    writer = None
    # if checkpoints is None:
    #     writer = SummaryWriter()
    if val:
        dl_test = dl_val
        hard_test_lines = hard_val_lines

    trainer = trainer_type(model, optimizer=None, scheduler=None, device=device, **train_args)
    fit_res = trainer.test(dl_test, checkpoints=checkpoints, writer=writer)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # fit_res = trainer.test(dl_train, checkpoints=checkpoints, writer=writer)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    save_experiment(run_name, out_dir, cfg, fit_res)
    if hard_test_lines is not None and len(hard_test_lines) > 0:
        if hasattr(trainer, 'save_results') and trainer.save_results is not None:
            trainer.save_results += '_hard'
        if hasattr(trainer, 'save_likelihoods') and trainer.save_likelihoods is not None:
            trainer.save_likelihoods += '_hard'
        if attribution_map is not None:
            data_args['attribution_map'] = attribution_paths[1]
        if 'mnli' in data_dir_prefix and save_results is not None:
            trainer._get_ids_for_mnli('other/mnli_hard_ids.csv')
            trainer.index=0
        if not val:
            ds_hard_test = dataset(hard_test_lines, hard_test_labels, tokenizer, max_len=max_len, **data_args)
        else:
            ds_hard_test = dataset(hard_val_lines, hard_val_labels, tokenizer, max_len=max_len, **data_args)
        if batches > 0:
            ds_test = Subset(ds_hard_test, range(batches * bs_test))
        dl_hard_test = torch.utils.data.DataLoader(ds_hard_test, bs_test, shuffle=False, **dataloader_args)
        fit_res = trainer.test(dl_hard_test, checkpoints=checkpoints, writer=writer)
        save_experiment(run_name + '_hard', out_dir, cfg, fit_res)

    del model
    if hyp is not None:
        del hyp
    torch.cuda.empty_cache()


def generate_dataset(data_dir_prefix='./data/snli_1.0/cl_snli_train', bs_test=8,
                   model_path=None, model_name='sshleifer/distilbart-cnn-12-6', model_type='bart', save_results=None, generate_all_labels=False,
                   ## generation params
                #    beam_size=4, max_length=32, early_stopping=True,
                   ):

    if save_results is None:
        save_results = data_dir_prefix + '_generated'
    else:
        save_results = save_results

    if os.path.isfile(save_results+'_lbl_file'):
        raise argparse.ArgumentError(f"File {save_results} Already exist!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(data_dir_prefix + '_lbl_file') as labels_file:
        labels = labels_file.readlines()
    with open(data_dir_prefix + '_source_file') as lines_file:
        lines = lines_file.readlines()

    model = get_model(model=model_type, model_name=model_name, model_path=model_path)
    model.config.min_length = 5
    model.config.max_length = 32
    model.config.task_specific_params['summarization']['min_length'] = 5
    model.config.task_specific_params['summarization']['max_length'] = 32
    # import pdb; pdb.set_trace()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    size_test = 10 ** 8

    all_labels_text = list(set(labels[:size_test]))
    all_labels_text.sort()
    num_labels = len(all_labels_text)
    all_labels = ['[' + l.upper().replace('\n', '') + ']' for l in all_labels_text]

    tokenizer.add_tokens(all_labels)
    labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]
    print(f'Labels IDs: {labels_ids}')

    dataset = DiscriminativeDataset(lines, labels, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, bs_test, shuffle=False)

    trainer = GenerativeTrainer(model=model, optimizer=None, scheduler=None, possible_labels_ids=labels_ids, 
                                tokenizer_encoder=tokenizer, save_results=save_results, device=device, generate_all_labels=generate_all_labels)
    trainer.generate_dataset(dataloader)


def pipeline(run_name, hyp_only_model=None, model_name='facebook/bart-base', train_hyp=False, seed=None, attribution_map=None,
                data_dir_prefix='./data/snli_1.0/cl_snli',word_dropout=0.0,weight_decay=0.0, hard_validation=False, test_with_prior=False,
                ft_epochs=20):
    lr=1e-5
    bs_train = 8
    bs_test = 8
    if 'large' in model_name or 'mnli' in data_dir_prefix:
        bs_train = 4
        bs_test = 4
    if 'bart' in model_name:
        model_type = 'bart'
    elif 'gpt' in model_name:
        model_type = 'decoder-only'
    else:
        model_type = 'encode-decode'
    checkpoints = f'checkpoints/{run_name}'

    ## if there is no hypothesis-only model, train one from scratch
    if hyp_only_model is None:
        print("********************** Training p(y|H) model **********************")
        hyp_run_name = f'{run_name}_hyp_only'
        hyp_checkpoints = f'checkpoints/{hyp_run_name}'
        run_experiment(hyp_run_name, data_dir_prefix=data_dir_prefix,
                   model_name=model_name, model_type='disc', seed=seed, 
                   checkpoints=hyp_checkpoints, lr=lr, hypothesis_only=True)
        hyp_only_model = f'{hyp_run_name}_model'
    else:
        print("********************** Using pre-trained p(y|H) model **********************")
    torch.cuda.empty_cache()


    model_path = f'{checkpoints}_model'
    if not os.path.isdir(model_path):
        ## Train p(P|y,H)*p(y|H) 
        print("********************** Training p(P|y,H)*p(y|H) model **********************")
        run_experiment(run_name, data_dir_prefix=data_dir_prefix, model_name=model_name, model_type=model_type, seed=seed, bs_train=bs_train, bs_test=bs_test,
                    checkpoints=checkpoints, lr=lr, word_dropout=word_dropout, hyp_only_model=hyp_only_model, hard_validation=False, 
                    weight_decay=weight_decay, attribution_map=attribution_map, train_hyp=train_hyp, test_with_prior=True)
    else:
        print("********************** Using pre-trained p(P|y,H) model **********************")
    

    print("********************** Testing p(P|y,H) model **********************")
    test_model(run_name=run_name, out_dir="results_test", data_dir_prefix=data_dir_prefix, model_name=model_name, bs_test=bs_test,
                model_path=model_path, model_type=model_type, seed=seed, attribution_map=attribution_map, test_with_prior=False)

    if train_hyp:
        hyp_only_model = f'{model_path}_disc_prior'

    ft_run_name = f'{run_name}_ft'
    ft_checkpoints = f'checkpoints/{ft_run_name}'
    ft_lr = lr / 2
    # bs_train = 4 if 'large' in model_name else bs_train
    # bs_test = 4 if 'large' in model_name else bs_test

    print("********************** Fine-tuning model (p(P|y,H)*p(y|H)) / (sum y' of p(P|y',H)*p(y'|H)) **********************")
    run_experiment(ft_run_name, data_dir_prefix=data_dir_prefix, bs_train=bs_train, bs_test=bs_test, model_name=model_name, 
    model_type=model_type, model_path=model_path, seed=seed, checkpoints=ft_checkpoints, lr=ft_lr, word_dropout=word_dropout, 
    hyp_only_model=hyp_only_model, hard_validation=hard_validation, weight_decay=weight_decay, attribution_map=attribution_map, 
    gamma=1.0, test_with_prior=test_with_prior, epochs=ft_epochs)
    ft_model_path = f'{ft_checkpoints}_model'

    print("********************** Testing fine-tuned model **********************")
    test_model(run_name=ft_run_name, out_dir="results_test", data_dir_prefix=data_dir_prefix, model_name=model_name, bs_test=bs_test,
                model_path=ft_model_path, model_type=model_type, seed=seed, attribution_map=attribution_map, test_with_prior=False)

    print("********************** Finished **********************")

    


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
    sp_exp.add_argument('--gen-premise', type=str,
                        default='', required=False)
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
    sp_exp.add_argument('--non-discriminative-bias', '-ndb', help='Make the synthetic bias non-discriminative', 
                        dest='non_discriminative_bias', action='store_true')
    sp_exp.add_argument('--attribution-map', '-am', type=str, 
                        help='path of attribution maps folder',
                        default=None)
    sp_exp.add_argument('--filt-method', '-fm', type=str, 
                        help='The method to filter the premis by. Should be in [sum,mean,max,max-abs,min-abs,true,rand',
                        default='none')
    sp_exp.add_argument('--move-to-hypothesis', '-mth', help='Move the filtered words from the premise to the hypothesis', 
                        dest='move_to_hypothesis', action='store_true')

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
    sp_exp.add_argument('--sched', type=str,
                        help='Which type of optimizer to use', default="linear")
    sp_exp.add_argument('--reduction', '-reduce', type=str,
                        help='How to reduce loss, can be "sum" or "mean"', default="sum")
    sp_exp.add_argument('--momentum', '-m', type=float,
                        help='Momentum for SGD', default=0.9)
    sp_exp.add_argument('--word-dropout', '-wdo', type=float,
                        help='Word dropout rate during training', default=0.0)
    sp_exp.add_argument('--label-smoothing-epsilon', '-lse', type=float,
                        help='Epsilon argument for label smoothing (does not uses labels smoothing by \
                        default', default=0.0)
    sp_exp.add_argument('--hyp-only-model', '-hom', type=str,
                        help='If you want to weigh loss by htpothesis only output', default=None)
    sp_exp.add_argument('--attribution-tokenizer', '-at', type=str,
                        help='Huggingface model name for the attributions, default is same as encoder', default=None)
    sp_exp.add_argument('--threshold', '-th', type=float, default=0.0)
    sp_exp.add_argument('--train-hyp', dest='train_hyp', action='store_true')
    sp_exp.add_argument('--test-with-prior', '-twp', dest='test_with_prior', action='store_true')
    sp_exp.add_argument('--cheat', dest='cheat', action='store_true')
    sp_exp.add_argument('--calc-uniform', '-cu', dest='calc_uniform', action='store_true')

    sp_exp.add_argument('--tie-embeddings', '-te', dest='tie_embeddings', action='store_true')
    sp_exp.add_argument('--hypothesis-only', '-ho', dest='hypothesis_only', action='store_true')
    sp_exp.add_argument('--premise-only', '-po', dest='premise_only', action='store_true')
    sp_exp.add_argument('--gradual-unfreeze', '-gu', dest='gradual_unfreeze', action='store_true')
    sp_exp.add_argument('--generate-hypothesis', '-gh', dest='generate_hypothesis', action='store_true')
    sp_exp.add_argument('--hard-validation', '-hv', dest='hard_validation', action='store_true')
    sp_exp.add_argument('--merge-train', dest='merge_train', action='store_true')
    sp_exp.add_argument('--label', '-l', type=int,
                        help='Create generative model only for one label', default=None)
    sp_exp.add_argument('--reverse' ,'-rev', dest='reverse', action='store_true', help='Generate hypothesis')
    sp_exp.add_argument('--tie-encoder-decoder', '-ted', dest='tie_encoder_decoder', action='store_true')
    sp_exp.add_argument('--pure-gen', '-pg', dest='pure_gen', action='store_true')

    sp_exp.set_defaults(tie_embeddings=False, hypothesis_only=False,
                        generate_hypothesis=False, non_discriminative_bias=False, gradual_unfreeze=False,
                        hard_validation=False, merge_train=False, train_hyp=False, test_with_prior=False,premise_only=False,
                        cheat=False, tie_encoder_decoder=False, calc_uniform=False, reverse=False, pure_gen=False)

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
    # sp_test.add_argument('--drive', '-d', type=bool, help='Pass "True" if you are running this on Google Colab',
    #                      default=False, required=False)
    sp_test.add_argument('--save-results', '-sr', type=str, help='Pass path if you want to save the results',
                         default=None, required=False)
    sp_test.add_argument('--reduction', '-reduce', type=str,
                        help='How to reduce loss, can be "sum" or "mean"', default="sum")
    sp_test.add_argument('--filt-method', '-fm', type=str, 
                        default='none')

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
    sp_test.add_argument('--threshold', '-th', type=float, default=0.0)
    sp_test.add_argument('--attribution-tokenizer', '-at', type=str,
                        help='Huggingface model name for the attributions, default is same as encoder', default=None)
    sp_test.add_argument('--move-to-hypothesis', '-mth', dest='move_to_hypothesis', action='store_true')
    sp_test.add_argument('--hyp-only-model', '-hom', type=str,
                        help='If you want to weigh loss by htpothesis only output', default=None)
    sp_test.add_argument('--test-with-prior', '-twp', dest='test_with_prior', action='store_true')
    sp_test.add_argument('--calc-uniform', '-cu', dest='calc_uniform', action='store_true')
    sp_test.add_argument('--reverse' ,'-rev', dest='reverse', action='store_true', help='Generate hypothesis')
    sp_test.add_argument('--inject-bias', type=int,
                        help='Select number of labels to inject bias to their corresponding hypotheses',
                        default=0)
    sp_test.add_argument('--bias-ids', type=int, nargs='+', help='Select the ids of the biases symbols',
                        default=[30000, 30001, 30002])
    sp_test.add_argument('--bias-ratio', type=float,
                        help='Select the percentege of labels to inject bias to their corresponding hypotheses',
                        default=0.5)
    sp_test.add_argument('--bias-location', type=str,
                        help='Select where in the hypotheses to inject the bias, can be either "start" or "end", otherwise will be random location',
                        default='start')
    sp_test.add_argument('--non-discriminative-bias', '-ndb', help='Make the synthetic bias non-discriminative', 
                        dest='non_discriminative_bias', action='store_true')
    sp_test.set_defaults(create_premises=False, move_to_hypothesis=False, test_with_prior=False, calc_uniform=False, reverse=False,
                        non_discriminative_bias=False)


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
    sp_test.add_argument('--premise-only', '-po', dest='premise_only', action='store_true')
    sp_test.add_argument('--pure-gen', '-pg', dest='pure_gen', action='store_true')
    sp_test.add_argument('--generate-hypothesis', '-gh', dest='generate_hypothesis', action='store_true')
    sp_test.add_argument('--val', '-val', dest='val', action='store_true')
    sp_test.set_defaults(hypothesis_only=False, generate_hypothesis=False, premise_only=False, pure_gen=False, val=False)
    sp_test.add_argument('--save-likelihoods', '-sl', type=str, help='Pass path if you want to save the likelihoods as a torch tensor',
                         default=None, required=False)

    sp_gen = sp.add_parser('generate', help='Generate new dataset')
    sp_gen.set_defaults(subcmd_fn=generate_dataset)
    sp_gen.add_argument('--data-dir-prefix', type=str,
                         help='Prefix of the path to data', default='./data/snli_1.0/cl_snli_train')
    sp_gen.add_argument('--model-path', '-mp', type=str,
                         help='Path of the first model', required=True)
    sp_gen.add_argument('--model-type', '-mt', type=str,
                         help='Type of the first model', default='bart')
    sp_gen.add_argument('--model-name', '-mn', type=str,
                         help='Name of the first model', default='sshleifer/distilbart-cnn-12-6')
    sp_gen.add_argument('--bs-test', '-bst', type=int,
                         help='Test batch size', default=8)
    sp_gen.add_argument('--save-results', '-sr', type=str, help='Pass path if you want to save the results',
                         default=None, required=False)
    sp_gen.add_argument('--generate-all-labels', '-gal', dest='generate_all_labels', action='store_true', help='Generate premises for all the labels and not just for gold labels')

    # Experiment config
    sp_pip = sp.add_parser('pipeline', help='Pipeline')
    sp_pip.set_defaults(subcmd_fn=pipeline)
    sp_pip.add_argument('--run-name', '-n', type=str,
                     help='Name of run and output file', required=True)
    sp_pip.add_argument('--seed', '-s', type=int, help='Random seed',
                     default=None, required=False)
    sp_pip.add_argument('--attribution-map', '-am', type=str, 
                     help='path of attribution maps folder', default=None)
    sp_pip.add_argument('--data-dir-prefix', type=str,
                     help='Prefix of the path to data', default='./data/snli_1.0/cl_snli')
    sp_pip.add_argument('--word-dropout', '-wdo', type=float,
                     help='Word dropout rate during training', default=0.0)
    sp_pip.add_argument('--hyp-only-model', '-hom', type=str,
                     help='If you want to weigh loss by htpothesis only output', default=None)
    sp_pip.add_argument('--train-hyp', dest='train_hyp', action='store_true')
    sp_pip.add_argument('--hard-validation', '-hv', dest='hard_validation', action='store_true')
    sp_pip.add_argument('--test-with-prior', '-twp', dest='test_with_prior', action='store_true')
    
    sp_pip.set_defaults(hard_validation=False, train_hyp=False)

    sp_pip.add_argument('--model-name', type=str,
                     help='Name of the huggingface model', default='facebook/bart-base')
    sp_pip.add_argument('--weight-decay', '-wd', type=float,
                        default=0.0)


    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed


# run_experiment("dimi", data_dir_prefix="../data/scitail/cl_scitail", bs_train=8, bs_test=4)

if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.empty_cache()
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    if torch.cuda.device_count() > 1:
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
