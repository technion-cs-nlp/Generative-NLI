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


class PremiseGenerationDataset(Dataset):

    def __init__(self, lines, labels, tokenizer_encoder, tokenizer_decoder=None, sep='|||', max_len=512,
                 dropout=0.0, generate_hypothesis=False):
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = labels
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder if tokenizer_decoder is not None else tokenizer_encoder
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)
        self.dropout = dropout
        self.generate_hypothesis = generate_hypothesis

    def __getitem__(self, index):

        split = self.lines[index].split(self.sep)

        premise = split[0]  # Premise
        hypothesis = split[1].replace('\n', '')  # Hypothesis

        if self.generate_hypothesis:
            hypothesis, premise = premise, hypothesis
        #
        # target_input = "<START> " + premise[:-1]

        sent = '[' + self.labels[index][:-1].upper().replace(' ', '') + '] ' + hypothesis

        input_dict = self.tokenizer_encoder.encode_plus(sent,
                                                        max_length=self.max_len,
                                                        pad_to_max_length=True,
                                                        return_tensors='pt',
                                                        truncation=True,
                                                        )
        target_dict = self.tokenizer_decoder.encode_plus(premise,
                                                         max_length=self.max_len,
                                                         pad_to_max_length=True,
                                                         return_tensors='pt',
                                                         truncation=True,
                                                         )

        # except Exception as e:
        #     print(e)
        #     print(premise)

        def create_mask(inp, tokenizer):
            mask = torch.FloatTensor(*inp.shape).uniform_() > self.dropout
            do_not_mask = (inp == tokenizer.pad_token_id)  ## do not mask paddings
            mask[do_not_mask] = True
            mask[:2] = True  # do not mask bos, label token

            if inp[-1] != tokenizer.pad_token_id:
                eos_loc = -1
            else:
                eos_loc = (do_not_mask).nonzero()[0][0] - 1
            mask[eos_loc] = True  # do not mask eos

            return mask

        if self.dropout > 0.0:  ## word dropout
            drop_inp = input_dict['input_ids'].squeeze(0)
            mask = create_mask(drop_inp, self.tokenizer_encoder)
            drop_inp = drop_inp * mask + self.tokenizer_encoder.unk_token_id * ~mask

            drop_out = target_dict['input_ids'].squeeze(0)
            mask = create_mask(drop_out, self.tokenizer_decoder)
            drop_out = drop_out * mask + self.tokenizer_decoder.unk_token_id * ~mask

            res = (drop_inp, input_dict['attention_mask'].squeeze(0)
                   , drop_out, target_dict['attention_mask'].squeeze(0))
        else:
            res = [input_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask']] + \
                  [target_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask']]

        # res = [torch.tensor(input_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']] + \
        #       [torch.tensor(target_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']]

        return tuple(res)

    def __len__(self):
        return self.size


class DiscriminativeDataset(Dataset):

    def __init__(self, lines, labels, tokenizer=None, sep='|||', max_len=512, dropout=0.0,
                 inject_bias=0, bias_ids=None, bias_ratio=0.5, bias_location='start',
                 non_discriminative_bias=False, seed=42, threshold=-100.0, 
                 attribution_map=None, move_to_hypothesis=False, filt_method='true',
                 attribution_tokenizer=None):
        if bias_ids is None:
            bias_ids = [2870, 2874, 2876]
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = self._labels_to_idx(labels)
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
        self.num_labels = len(list(set(labels)))
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

        self.alpha = 0.25

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
        labels_ids = list(set(labels))
        labels_ids.sort()
        res = [labels_ids.index(label) for label in labels]

        return res

    def __getitem__(self, index):

        split = self.lines[index].split(self.sep)

        premise = split[0]
        premise_len = len(premise.split(' '))
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
            # premise_splited = premise.split()
            # premise_splited = [(word if np.random.random() > self.dropout else self.tokenizer.unk_token)
            #                    for word in premise_splited]
            # premise = ' '.join(premise_splited)

            hypothesis_splited = hypothesis.split()
            hypothesis_splited = [(word if np.random.random() > self.dropout else self.tokenizer.unk_token)
                                  for word in hypothesis_splited]
            hypothesis = ' '.join(hypothesis_splited)

        if self.inject_bias > lbl:
            with temp_seed(index):  # stay the same for every epoch
                rand = np.random.random()
                delta = (1 - self.bias_ratio) / (self.num_labels - 1)
                bias_idx = None
                if rand < self.bias_ratio:  # randomly add bias
                    bias_idx = lbl.item()
                elif self.non_discriminative_bias:
                    lower_bound = self.bias_ratio
                    upper_bound = self.bias_ratio + delta
                    for i in range(1, len(self.bias_ids)):
                        if lower_bound <= rand < upper_bound:
                            bias_idx = (lbl.item() + i) % self.num_labels
                            break
                        lower_bound = upper_bound
                        upper_bound = lower_bound + delta

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

    def __init__(self, lines, labels, tokenizer=None, sep='|||', max_len=512, dropout=0.0, **kw):
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = self._labels_to_idx(labels)
        self.tokenizer = tokenizer
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)

    def _labels_to_idx(self, labels):
        labels_ids = list(set(labels))
        labels_ids.sort()
        res = [labels_ids.index(label) for label in labels]

        return res

    def __getitem__(self, index):
        split = self.lines[index].split(self.sep)

        # premise = split[0]
        hypothesis = split[1].replace('\n', '')
        lbl = torch.tensor(self.labels[index])

        return (hypothesis, lbl)

    def __len__(self):
        return self.size


class DualDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        super().__init__()
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        return self.datasetA[index], self.datasetB[index]

    def __len__(self):
        return len(self.datasetA)
