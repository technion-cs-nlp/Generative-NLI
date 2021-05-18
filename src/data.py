import pdb
import torch
from torch.utils.data import Dataset
import numpy as np
import contextlib
from collections import defaultdict


@contextlib.contextmanager
def temp_state(state_t):
    state = np.random.get_state()
    np.random.set_state(state_t)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class DiscriminativeDataset(Dataset):

    def __init__(self, lines, labels, tokenizer=None, sep='|||', max_len=512, dropout=0.0,
                 inject_bias=0, bias_ids=None, bias_ratio=0.5, bias_location='start',
                 non_discriminative_bias=False, misalign_bias=False, seed=42, threshold=0.0, 
                 attribution_map=None, move_to_hypothesis=False, filt_method='true',
                 attribution_tokenizer=None, possible_labels=None, rev=False, pure_gen=False, hypothesis_only=False, premise_only=False):
        if bias_ids is None:
            bias_ids = [2870, 2874, 2876]
        super().__init__()
        self.lines = lines
        self.possible_labels = possible_labels
        if labels is not None:
            self.labels = self._labels_to_idx(labels)
            assert len(lines) == len(labels)
        else:
            self.labels = None
        self.tokenizer = tokenizer
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)
        self.dropout = dropout
        self.inject_bias = inject_bias
        self.bias_ids = bias_ids
        self.bias_ratio = bias_ratio
        self.bias_location = bias_location
        self.non_discriminative_bias = non_discriminative_bias
        self.misalign_bias = misalign_bias
        self.num_labels = len(list(set(labels))) if labels is not None else 3
        self.threshold = threshold
        self.attribution_map = attribution_map
        if attribution_map is not None:
            if attribution_tokenizer is None:
                self.tokenizer_attr = tokenizer
            else:
                from transformers import AutoTokenizer
                self.tokenizer_attr = AutoTokenizer.from_pretrained(attribution_tokenizer)
        self.move_to_hypothesis = move_to_hypothesis
        self.filt_method = filt_method
        self.rev = rev
        self.alpha = 0.25
        self.pure_gen = pure_gen
        self.premise_only = premise_only
        self.hypothesis_only = hypothesis_only

    def _create_hist(self):
        hist = defaultdict(lambda: defaultdict(int))
        for label, line in zip(self.labels, self.lines):
            premise, hypothesis = line.split(self.sep)
            for word in hypothesis.strip().lower().split():
                hist['total'][word] += 1
                hist[label][word] += 1

        # import pdb; pdb.set_trace()

        return hist

    def _labels_to_idx(self, labels):
        # import pdb;pdb.set_trace()
        if self.possible_labels is None:
            self.possible_labels = list(set(labels))
        self.possible_labels.sort()
        res = [(self.possible_labels.index(l) if l in self.possible_labels else 0) for l in labels]

        return res

    def __getitem__(self, index):
        # if type(self.lines) == list:
        split = self.lines[index].split(self.sep)

        premise = split[0]
        hypothesis = split[1].replace('\n', '')
        lbl = torch.tensor(self.labels[index])

        if self.attribution_map is not None and self.attribution_map[index] is not None:
            threshold = self.threshold
            if type(self.attribution_map[index])!=list:
                # import pdb; pdb.set_trace()
                premise, hypothesis = self.filter_premise(premise, hypothesis, self.attribution_map[index], threshold)

            elif None not in self.attribution_map[index]:
                # import pdb; pdb.set_trace()
                if self.filt_method == 'sum':
                    filt = torch.stack(self.attribution_map[index]).sum(0)
                elif self.filt_method == 'max':
                    filt = torch.stack(self.attribution_map[index]).max(0).values
                elif self.filt_method == 'min-abs':
                    filt = torch.stack(self.attribution_map[index]).abs().min(0).values
                elif self.filt_method == 'max-abs':
                    # import pdb; pdb.set_trace()
                    filt = torch.stack(self.attribution_map[index]).abs().max(0).values
                elif self.filt_method == 'true':
                    filt = self.attribution_map[index][lbl]
                elif self.filt_method == 'mean':
                    filt = torch.stack(self.attribution_map[index]).sum(0) / len(self.attribution_map[index])
                elif self.filt_method == 'rand':
                    if np.random.random() > 0.5:
                        filt = torch.stack(self.attribution_map[index]).abs().max(0).values
                    else:
                        threshold = 0.0
                        filt = self.attribution_map[index][lbl]
                else: # self.filt_method == 'none'
                    premises, hypotheses = [], []
                    for filt in self.attribution_map[index]:
                        P, H = self.filter_premise(premise, hypothesis, filt, threshold)
                        premises.append(P)
                        hypotheses.append(H)
                    return premises, hypotheses, lbl

                
                premise, hypothesis = self.filter_premise(premise, hypothesis, filt, threshold)

        if self.dropout > 0.0:

            hypothesis_splited = hypothesis.split()
            hypothesis_splited = [(word if np.random.random() > self.dropout else self.tokenizer.unk_token)
                                  for word in hypothesis_splited]
            hypothesis = ' '.join(hypothesis_splited)

        if self.inject_bias > lbl:
            with temp_seed(index):  # stay the same for every epoch
                rand = np.random.random()
                delta = (1 - self.bias_ratio) / (self.num_labels)
                bias_idx = None
                if rand <= self.bias_ratio:  # randomly add bias
                    bias_idx = lbl.item()
                elif self.non_discriminative_bias:
                    idx = np.random.choice(list(range(self.num_labels)))
                    bias_idx = (lbl.item() + idx) % self.num_labels

                if self.misalign_bias:
                    bias_idx = (bias_idx + 1) % self.num_labels

                if bias_idx is not None:
                    hypothesis_splited = hypothesis.split()
                    if self.bias_location == 'start':
                        idx = 0
                    elif self.bias_location == 'end':
                        idx = 500
                    else:  # random location
                        idx = np.random.randint(len(hypothesis_splited))
                    # import pdb; pdb.set_trace()
                    bias_str = self.tokenizer.decode(self.bias_ids[bias_idx]).replace(' ', '')
                    hypothesis_splited = hypothesis_splited[0:idx] + [bias_str] + hypothesis_splited[idx:]
                    hypothesis = ' '.join(hypothesis_splited)
        if self.rev:
            return hypothesis, premise, lbl
        elif self.pure_gen:
            return f'{premise}{self.tokenizer.sep_token}{hypothesis}', lbl
        elif self.hypothesis_only:
            # import pdb; pdb.set_trace()
            return hypothesis, lbl
        return premise, hypothesis, lbl  # P, H, y

    def filter_premise(self, premise, hypothesis, filt, threshold):
        premise_encoded = self.tokenizer_attr(premise,return_tensors='pt').input_ids.view(-1)
        premise_len = len(premise_encoded)
        premise_attr = filt.view(-1)[:premise_len]
        premise_attr_normal = premise_attr # / premise_attr.sum()
        mask = premise_attr_normal >= threshold
        # import pdb; pdb.set_trace()
        premise_encoded_filtered = premise_encoded[mask]
        premise = self.tokenizer_attr.decode(premise_encoded_filtered,skip_special_tokens=True)
        if self.move_to_hypothesis:
            premise_encoded_dropped = premise_encoded[~mask]
            dropped = self.tokenizer_attr.decode(premise_encoded_dropped,skip_special_tokens=True)
            # import pdb; pdb.set_trace()
            hypothesis = f"{dropped} {self.tokenizer_attr.sep_token} {hypothesis}"
        
        return premise, hypothesis

    def __len__(self):
        return self.size


class HypothesisOnlyDataset(Dataset):

    def __init__(self, lines, labels=None, tokenizer=None, sep='|||', max_len=512, dropout=0.0, premise_only=False, 
                    threshold=0.0, attribution_map=None, move_to_hypothesis=False, filt_method='true',
                    attribution_tokenizer=None, possible_labels=None, **kw):
        super().__init__()
        self.lines = lines
        self.possible_labels=possible_labels
        if labels is not None:
            self.labels = self._labels_to_idx(labels)
            assert len(lines) == len(labels)
        else:
            self.labels = None
        self.tokenizer = tokenizer
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)
        self.premise_only = premise_only
        self.attribution_map = attribution_map
        self.move_to_hypothesis = move_to_hypothesis
        self.threshold = threshold
        self.filt_method = filt_method
        if attribution_map is not None:
            if attribution_tokenizer is None:
                self.tokenizer_attr = tokenizer
            else:
                from transformers import AutoTokenizer
                self.tokenizer_attr = AutoTokenizer.from_pretrained(attribution_tokenizer)


    def _labels_to_idx(self, labels):
        # import pdb;pdb.set_trace()
        if self.possible_labels is None:
            self.possible_labels = list(set(labels))
        self.possible_labels.sort()
        res = [(self.possible_labels.index(l) if l in self.possible_labels else 0) for l in labels]
        return res

    def __getitem__(self, index):
        if type(self.lines) == list:
            split = self.lines[index].split(self.sep)

            premise = split[0]
            hypothesis = split[1].replace('\n', '') if len(split)>1 else premise.strip()
            lbl = torch.tensor(self.labels[index])
        else:
            premise = self.lines[index]["evidence"]
            hypothesis = self.lines[index]["claim"]
            lbl = torch.tensor(self.lines[index]["label"])

        if self.attribution_map is not None and self.attribution_map[index] is not None:
            threshold = self.threshold
            if type(self.attribution_map[index])!=list:
                # import pdb; pdb.set_trace()
                premise, hypothesis = self.filter_premise(premise, hypothesis, self.attribution_map[index], threshold)

            elif None not in self.attribution_map[index]:
                # import pdb; pdb.set_trace()
                if self.filt_method == 'sum':
                    filt = torch.stack(self.attribution_map[index]).sum(0)
                elif self.filt_method == 'max':
                    filt = torch.stack(self.attribution_map[index]).max(0).values
                elif self.filt_method == 'min-abs':
                    filt = torch.stack(self.attribution_map[index]).abs().min(0).values
                elif self.filt_method == 'max-abs':
                    # import pdb; pdb.set_trace()
                    filt = torch.stack(self.attribution_map[index]).abs().max(0).values
                elif self.filt_method == 'true':
                    filt = self.attribution_map[index][lbl]
                elif self.filt_method == 'mean':
                    filt = torch.stack(self.attribution_map[index]).sum(0) / len(self.attribution_map[index])
                elif self.filt_method == 'rand':
                    if np.random.random() > 0.5:
                        filt = torch.stack(self.attribution_map[index]).abs().max(0).values
                    else:
                        threshold = 0.0
                        filt = self.attribution_map[index][lbl]
                else: # self.filt_method == 'none'
                    premises, hypotheses = [], []
                    for filt in self.attribution_map[index]:
                        P, H = self.filter_premise(premise, hypothesis, filt, threshold)
                        premises.append(P)
                        hypotheses.append(H)
                    return premises, hypotheses, lbl

                
                premise, hypothesis = self.filter_premise(premise, hypothesis, filt, threshold)
                
        return (hypothesis, lbl) if not self.premise_only else (premise, lbl)

    def __len__(self):
        return self.size

    def filter_premise(self, premise, hypothesis, filt, threshold):
        premise_encoded = self.tokenizer_attr(premise,return_tensors='pt').input_ids.view(-1)
        premise_len = len(premise_encoded)
        premise_attr = filt.view(-1)[:premise_len]
        premise_attr_normal = premise_attr # / premise_attr.sum()
        mask = premise_attr_normal >= threshold
        premise_encoded_filtered = premise_encoded[mask]
        premise = self.tokenizer_attr.decode(premise_encoded_filtered,skip_special_tokens=True)
        if self.move_to_hypothesis:
            premise_encoded_dropped = premise_encoded[~mask]
            dropped = self.tokenizer_attr.decode(premise_encoded_dropped,skip_special_tokens=True)
            hypothesis = f"{dropped} {self.tokenizer_attr.sep_token} {hypothesis}"
        
        return premise, hypothesis


