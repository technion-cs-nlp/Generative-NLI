import os
import sys

import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer,BartTokenizer, BertForQuestionAnswering, BertConfig, BertForSequenceClassification, BartForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from transformers.utils.dummy_pt_objects import torch_distributed_zero_first

from src.data import PremiseGenerationDataset, DiscriminativeDataset, HypothesisOnlyDataset, DualDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'bert-base-uncased'
# # model_path = 'checkpoints/exp_b2b_mnli_disc_model'
model_path = 'checkpoints/exp_b2b_mnli_disc_model'
# model_name = 'facebook/bart-base'
# model_path = 'checkpoints/exp_bart_b_disc_model'
prefix = 'mnli'
labels_type = 'true'
name = "hard_test_set"

# load model
model = AutoModelForSequenceClassification.from_pretrained(model_path, return_dict=False)
# model_cpu = BertForSequenceClassification.from_pretrained(model_path, return_dict=False)
# model_cpu.eval()
# model_cpu.zero_grad()
model.to(device)
model.eval()
model.zero_grad()

# model = model_device

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# create dataset
# data_dir_prefix = './data/snli_1.0/cl_snli'
data_dir_prefix = './data/mnli/cl_multinli'
test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')
val_str = ('dev_matched' if 'mnli' in data_dir_prefix else 'val')

with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
    test_labels = test_labels_file.readlines()
with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
    test_lines = test_lines_file.readlines()
with open(data_dir_prefix + f'_train_lbl_file') as train_labels_file:
    train_labels = train_labels_file.readlines()
with open(data_dir_prefix + f'_train_source_file') as train_lines_file:
    train_lines = train_lines_file.readlines()
with open(data_dir_prefix + f'_{val_str}_lbl_file') as val_labels_file:
    val_labels = val_labels_file.readlines()
with open(data_dir_prefix + f'_{val_str}_source_file') as val_lines_file:
    val_lines = val_lines_file.readlines()
# import pdb; pdb.set_trace()
if os.path.isfile(data_dir_prefix + f'_{test_str}_hard_lbl_file') and \
        os.path.isfile(data_dir_prefix + f'_{test_str}_hard_source_file'):
    with open(data_dir_prefix + f'_{test_str}_hard_lbl_file') as hard_test_labels_file:
        hard_test_labels = hard_test_labels_file.readlines()
    with open(data_dir_prefix + f'_{test_str}_hard_source_file') as hard_test_lines_file:
        hard_test_lines = hard_test_lines_file.readlines()

if 'train_set' in name:
    dataset = DiscriminativeDataset(lines=train_lines, labels=train_labels)
elif 'val_set' in name:
    dataset = DiscriminativeDataset(lines=val_lines, labels=val_labels)
elif 'test_set' in name and 'hard' not in name:
    dataset = DiscriminativeDataset(lines=test_lines, labels=test_labels)
elif 'hard_test_set' in name:
    dataset = DiscriminativeDataset(lines=hard_test_lines, labels=hard_test_labels)


def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    # import pdb; pdb.set_trace()
    args = {
        'attention_mask':attention_mask
    }
    if 'bart' not in model_name:
        args['token_type_ids']=token_type_ids
        args['position_ids']=position_ids
    return model(inputs, **args)


def squad_pos_forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
    pred = predict(inputs,
                   token_type_ids=token_type_ids,
                   position_ids=position_ids,
                   attention_mask=attention_mask)
    pred = pred[position]
    return pred


ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence


def construct_input_ref_pair(premise, hypothesis, ref_token_id, sep_token_id, cls_token_id):
    premise_ids = tokenizer.encode(premise, add_special_tokens=False)
    hypothesis_ids = tokenizer.encode(hypothesis, add_special_tokens=False)

    # construct input token ids
    input_ids = [cls_token_id] + premise_ids + [sep_token_id] + ([cls_token_id] if 'bart' in model_name else []) + \
                hypothesis_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(premise_ids) + [sep_token_id] + ([cls_token_id] if 'bart' in model_name else []) + \
                    [ref_token_id] * len(hypothesis_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(premise_ids)


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


def construct_bert_sub_embedding(input_ids, ref_input_ids,
                                 token_type_ids, ref_token_type_ids,
                                 position_ids, ref_position_ids):
    input_embeddings = interpretable_embedding1.indices_to_embeddings(input_ids)
    ref_input_embeddings = interpretable_embedding1.indices_to_embeddings(ref_input_ids)

    input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(token_type_ids)
    ref_input_embeddings_token_type = interpretable_embedding2.indices_to_embeddings(ref_token_type_ids)

    input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(position_ids)
    ref_input_embeddings_position_ids = interpretable_embedding3.indices_to_embeddings(ref_position_ids)

    return (input_embeddings, ref_input_embeddings), \
           (input_embeddings_token_type, ref_input_embeddings_token_type), \
           (input_embeddings_position_ids, ref_input_embeddings_position_ids)


def construct_whole_bert_embeddings(input_ids, ref_input_ids,
                                    token_type_ids=None, ref_token_type_ids=None,
                                    position_ids=None, ref_position_ids=None):
    input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids, token_type_ids=token_type_ids,
                                                                     position_ids=position_ids)
    ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids, token_type_ids=token_type_ids,
                                                                         position_ids=position_ids)

    return input_embeddings, ref_input_embeddings


# premise, hypothesis = "A woman with a green headscarf , blue shirt and a very big grin .", "The woman is very happy ."

def custom_forward(inputs, token_type_ids=None, position_ids=None, attention_mask=None, true_label=None):
    args = {
        'attention_mask':attention_mask
    }
    if 'bart' not in model_name:
        args['token_type_ids']=token_type_ids
        args['position_ids']=position_ids
    outputs = predict(inputs, **args)
    preds = outputs[0]  # logits
    # import pdb; pdb.set_trace()
    if true_label is None:
        return preds.max(1).values
    return preds[:,true_label]


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# import pdb; pdb.set_trace()
all_attributions = []
import sys
import tqdm

# attr_map = {}

with tqdm.tqdm(desc='Saving...', total=len(dataset),
               file=sys.stdout) as pbar:
    for premise, hypothesis, label in dataset:
        if model.device != device:
            torch.cuda.empty_cache()
            model.to(device)
        input_ids, ref_input_ids, sep_id = construct_input_ref_pair(premise, hypothesis, ref_token_id, sep_token_id,
                                                                    cls_token_id)
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
        position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
        attention_mask = construct_attention_mask(input_ids)

        # indices = input_ids[0].detach().tolist()
        # all_tokens = tokenizer.convert_ids_to_tokens(indices)

        # calculate attributes
        embeddings = None
        if 'roberta' in model_name:
            embeddings = model.roberta.embeddings
        elif 'bart' in model_name:
            embeddings = model.model.encoder.embed_tokens
        elif 'bert' in model_name:
            embeddings = model.bert.embeddings
        lig = LayerIntegratedGradients(custom_forward, embeddings)
        # pred_label = 'temp'
        if labels_type=='pred':
            with torch.no_grad():
                outputs = predict(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
                preds = outputs[0]  # logits
            pred_label = preds.max(1).indices[0].item()
        # # import pdb; pdb.set_trace()
        # probs = torch.nn.Softmax(0)(preds[0])
        temp = []
        
        # for l in range(3):
        n_steps=100
        try:
            # raise RuntimeError
            attributions, delta = lig.attribute(inputs=input_ids,
                                                baselines=ref_input_ids,
                                                additional_forward_args=(token_type_ids, position_ids, attention_mask,
                                                label if labels_type=='true' else pred_label),
                                                                        # pred_label),
                                                                        # l),
                                                # revise this
                                                return_convergence_delta=True,
                                                # More steps
                                                n_steps=n_steps
                                                )
        except RuntimeError as e:
            model.to(torch.device('cpu'))
            torch.cuda.empty_cache()
            # model = model_cpu
            attributions, delta = lig.attribute(inputs=input_ids.to(torch.device('cpu')),
                                baselines=ref_input_ids.to(torch.device('cpu')),
                                additional_forward_args=(token_type_ids.to(torch.device('cpu')), position_ids.to(torch.device('cpu')), 
                                                        attention_mask.to(torch.device('cpu')),
                                                        label if labels_type=='true' else pred_label),
                                                        # pred_label),
                                                        # l),
                                # revise this
                                return_convergence_delta=True,
                                # More steps
                                n_steps=n_steps
                                )
            torch.cuda.empty_cache()
            # try:
            #     model.to(device)
            # except RuntimeError as e:
            #     del model
            #     torch.cuda.empty_cache()
            #     model = BertForSequenceClassification.from_pretrained(model_path, return_dict=False)
            #     model.to(device)
            #     model.eval()
            #     model.zero_grad()
            # model = model_device

        attributions_sum = summarize_attributions(attributions) if attributions is not None else None
        if attributions_sum is not None:
            attributions_sum = attributions_sum.cpu()
        if 'bart' in model_name:
            attributions_sum = attributions_sum[1:]
        # ind_end = (ref_input_ids[0]==102).nonzero(as_tuple=False)[0][0]
        # attributions_sum = attributions_sum[:ind_end+1]
        # weighted_attributions = attributions_sum * probs[l]
        # temp.append(attributions_sum)
        # temp.append(weighted_attributions)

        # import pdb; pdb.set_trace()
        # ratios = torch.nn.Softmax(1)(outputs[0])[0]
        # temp_avg = sum([temp[i]*ratios[i] for i in range(3)]) if None not in temp else None
        # all_attributions.append(temp)
        all_attributions.append(attributions_sum)
        pbar.update()

if not os.path.isdir(f'attributions_{prefix}_{labels_type}'):
    os.mkdir(f'attributions_{prefix}_{labels_type}')
torch.save(all_attributions, f'attributions_{prefix}_{labels_type}/{name}.torch')
