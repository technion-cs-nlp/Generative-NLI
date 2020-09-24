import torch
from torch.utils.data import Dataset
import numpy as np
import contextlib

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

        premise = split[0]                      # Premise
        hypothesis = split[1].replace('\n', '')    # Hypothesis

        if self.generate_hypothesis:
            # import pdb; pdb.set_trace()
            hypothesis, premise = premise, hypothesis
        #
        # target_input = "<START> " + premise[:-1]

        sent = '[' + self.labels[index][:-1].upper().replace(' ', '') + '] ' + hypothesis

        # import pdb; pdb.set_trace()

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
        # import pdb; pdb.set_trace()
        def create_mask(inp, tokenizer):
            mask = torch.FloatTensor(*inp.shape).uniform_() > self.dropout
            do_not_mask = (inp==tokenizer.pad_token_id)  ## do not mask paddings
            mask[do_not_mask] = True
            mask[:2] = True      # do not mask bos, label token
            
            if inp[-1] != tokenizer.pad_token_id:
                eos_loc = -1
            else:
                eos_loc =(do_not_mask).nonzero()[0][0] - 1
            mask[eos_loc] = True     # do not mask eos
            
            return mask

        if self.dropout > 0.0:          ## word dropout
            drop_inp = input_dict['input_ids'].squeeze(0)
            mask = create_mask(drop_inp, self.tokenizer_encoder)
            drop_inp = drop_inp * mask + self.tokenizer_encoder.unk_token_id * ~mask

            drop_out = target_dict['input_ids'].squeeze(0)
            mask = create_mask(drop_out, self.tokenizer_decoder)
            drop_out = drop_out * mask + self.tokenizer_decoder.unk_token_id * ~mask

            res = (drop_inp,input_dict['attention_mask'].squeeze(0)
                    ,drop_out,target_dict['attention_mask'].squeeze(0))
        else:
            res = [input_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask']] + \
                [target_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask']]

        # res = [torch.tensor(input_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']] + \
        #       [torch.tensor(target_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']]

        return tuple(res)

    def __len__(self):
        return self.size


class DiscriminativeDataset(Dataset):

    def __init__(self, lines, labels, tokenizer, sep='|||', max_len=512, dropout=0.0, 
                inject_bias=0, bias_ids=30000, bias_ratio=0.5, bias_location='start', seed=42):
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
        with temp_seed(seed):
            self.state = np.random.get_state()
    
    def _labels_to_idx(self,labels):
        labels_ids = list(set(labels))
        labels_ids.sort()
        res = [labels_ids.index(label) for label in labels]
        
        return res

    def __getitem__(self, index):

        split = self.lines[index].split(self.sep)

        premise = split[0]
        hypothesis = split[1].replace('\n', '')
        lbl = torch.tensor(self.labels[index])

        if self.inject_bias > lbl:
            with temp_seed(index):                                  # stay the same for every epoch
                if np.random.random() < self.bias_ratio:            # randomally add bias
                    hypothesis_splited = hypothesis.split()
                    if self.bias_location == 'start':
                        idx = 0
                    elif self.bias_location == 'end':
                        idx = 500
                    else:       ## random location
                        idx = np.random.randint(len(hypothesis_splited))
                    # import pdb; pdb.set_trace()
                    bias_str = self.tokenizer.decode(self.bias_ids[lbl.item()]).replace(' ','')
                    hypothesis_splited = hypothesis_splited[0:idx] + [bias_str] + hypothesis_splited[idx:]
                    hypothesis = ' '.join(hypothesis_splited)    
            # import pdb; pdb.set_trace()     
        
        if self.dropout > 0.0:
            premise_splited = premise.split()
            premise_splited = [(word if np.random.random() > self.dropout else self.tokenizer.unk_token)
                                for word in premise_splited]
            premise = ' '.join(premise_splited)

            hypothesis_splited = hypothesis.split()
            hypothesis_splited = [(word if np.random.random() > self.dropout else self.tokenizer.unk_token)
                                for word in hypothesis_splited]
            hypothesis = ' '.join(hypothesis_splited)

        # import pdb; pdb.set_trace()
        return (premise,hypothesis,lbl)        # P, H, y

    def __len__(self):
        return self.size

class HypothesisOnlyDataset(Dataset):

    def __init__(self, lines, labels, tokenizer, sep='|||', max_len=512, dropout=0.0, **kw):
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = self._labels_to_idx(labels)
        self.tokenizer = tokenizer
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)

    def _labels_to_idx(self,labels):
        labels_ids = list(set(labels))
        labels_ids.sort()
        res = [labels_ids.index(label) for label in labels]
        
        return res

    def __getitem__(self, index):

        split = self.lines[index].split(self.sep)

        # premise = split[0]
        hypothesis = split[1].replace('\n', '')
        lbl = torch.tensor(self.labels[index])

        return (hypothesis,lbl)

    def __len__(self):
        return self.size


class DualDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        super().__init__()
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self,index):
        return self.datasetA[index], self.datasetB[index]

    def __len__(self):
        return len(self.datasetA)