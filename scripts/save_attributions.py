import os
import sys

import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig, BertForSequenceClassification

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from transformers.utils.dummy_pt_objects import torch_distributed_zero_first

from src.data import PremiseGenerationDataset, DiscriminativeDataset, HypothesisOnlyDataset, DualDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'bert-base-uncased'
model_path = 'checkpoints/bert_disc_model'

# load model
model = BertForSequenceClassification.from_pretrained(model_path, return_dict=False)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
# create dataset
data_dir_prefix = './data/snli_1.0/cl_snli'
val_str = ('dev_matched' if 'mnli' in data_dir_prefix else 'val')
test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')

with open(data_dir_prefix + '_train_lbl_file') as train_labels_file:
    train_labels = train_labels_file.readlines()
with open(data_dir_prefix + '_train_source_file') as train_lines_file:
    train_lines = train_lines_file.readlines()
with open(data_dir_prefix + f'_{val_str}_lbl_file') as val_labels_file:
    val_labels = val_labels_file.readlines()
with open(data_dir_prefix + f'_{val_str}_source_file') as val_lines_file:
    val_lines = val_lines_file.readlines()
with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
    test_labels = test_labels_file.readlines()
with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
    test_lines = test_lines_file.readlines()
if os.path.isfile(data_dir_prefix + '_test_hard_lbl_file') and \
        os.path.isfile(data_dir_prefix + '_test_hard_source_file'):
    with open(data_dir_prefix + '_test_hard_lbl_file') as val_labels_file:
        hard_test_labels = val_labels_file.readlines()
    with open(data_dir_prefix + '_test_hard_source_file') as val_lines_file:
        hard_test_lines = val_lines_file.readlines()

name = "train_set_set_true_100"

if 'train_set' in name:
    dataset = DiscriminativeDataset(lines=train_lines, labels=train_labels)
elif 'val_set' in name:
    dataset = DiscriminativeDataset(lines=val_lines, labels=val_labels)
elif 'test_set' in name and 'hard' not in name:
    dataset = DiscriminativeDataset(lines=test_lines, labels=test_labels)
elif 'hard_test_set' in name:
    dataset = DiscriminativeDataset(lines=hard_test_lines, labels=hard_test_labels)


def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    return model(inputs, token_type_ids=token_type_ids,
                 position_ids=position_ids, attention_mask=attention_mask, )


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
    input_ids = [cls_token_id] + premise_ids + [sep_token_id] + hypothesis_ids + [sep_token_id]

    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(premise_ids) + [sep_token_id] + \
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
    outputs = predict(inputs, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
    preds = outputs[0]  # logits
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

with tqdm.tqdm(desc='Saving...', total=len(dataset),
               file=sys.stdout) as pbar:
    for premise, hypothesis, label in dataset:
        input_ids, ref_input_ids, sep_id = construct_input_ref_pair(premise, hypothesis, ref_token_id, sep_token_id,
                                                                    cls_token_id)
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
        position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
        attention_mask = construct_attention_mask(input_ids)

        indices = input_ids[0].detach().tolist()
        all_tokens = tokenizer.convert_ids_to_tokens(indices)

        # calculate attributes
        lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)

        try:
            attributions, delta = lig.attribute(inputs=input_ids,
                                                baselines=ref_input_ids,
                                                additional_forward_args=(token_type_ids, position_ids, attention_mask,
                                                                        # label
                                                                        label),
                                                # revise this
                                                return_convergence_delta=True,
                                                # More steps
                                                n_steps=100
                                                )
        except RuntimeError as e:
            torch.cuda.empty_cache()
            try:
                attributions, delta = lig.attribute(inputs=input_ids,
                                                    baselines=ref_input_ids,
                                                    additional_forward_args=(token_type_ids, position_ids, attention_mask,
                                                                            # label
                                                                            label),
                                                    # revise this
                                                    return_convergence_delta=True,
                                                    # More steps
                                                    n_steps=50
                                                    )
            except Exception as e:
                attributions = None

        attributions_sum = summarize_attributions(attributions) if attributions is not None else None

        all_attributions.append(attributions_sum)
        pbar.update()


torch.save(all_attributions, f'attributions_100_true/{name}.torch')
