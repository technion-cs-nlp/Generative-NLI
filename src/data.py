import torch
from torch.utils.data import Dataset


class PremiseGenerationDataset(Dataset):

    def __init__(self, lines, labels, tokenizer_encoder, tokenizer_decoder=None, sep='|||', max_len=512):
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = labels
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder if tokenizer_decoder is not None else tokenizer_encoder
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)

    def __getitem__(self, index):

        split = self.lines[index].split(self.sep)

        inp = split[0]
        tgt = split[1].replace('\n', '')
        #
        # target_input = "<START> " + tgt[:-1]

        sent = '[' + self.labels[index][:-1].upper().replace(' ', '') + '] ' + inp

        # import pdb; pdb.set_trace()

        input_dict = self.tokenizer_encoder.encode_plus(sent,
                                                max_length=self.max_len,
                                                pad_to_max_length=True,
                                                return_tensors='pt',
                                                truncation=True,
                                                )
        target_dict = self.tokenizer_decoder.encode_plus(tgt,
                                                max_length=self.max_len,
                                                pad_to_max_length=True,
                                                return_tensors='pt',
                                                truncation=True,
                                                )
        # except Exception as e:
        #     print(e)
        #     print(tgt)
        #     import pdb; pdb.set_trace()

        res = [input_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask']] + \
              [target_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask']]

        # res = [torch.tensor(input_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']] + \
        #       [torch.tensor(target_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']]

        return tuple(res)

    def __len__(self):
        return self.size


class DiscriminitiveDataset(Dataset):

    def __init__(self, lines, labels, tokenizer, sep='|||', max_len=512):
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = _labels_to_idx(labels)
        self.tokenizer = tokenizer
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)

    @staticmethod    
    def _labels_to_idx(labels):
        labels_ids = list(set(labels))
        res = [labels_ids.index(label) for label in labels]
        
        return res

    def __getitem__(self, index):

        split = self.lines[index].split(self.sep)

        inp = split[0]
        tgt = split[1].replace('\n', '')
        lbl = torch.tensor(self.labels[index])

        # input_dict = self.tokenizer.encode_plus(inp, tgt,
        #                                         max_length=self.max_len,
        #                                         pad_to_max_length=True,
        #                                         return_tensors='pt',
        #                                         )

        # res = [input_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask', 'token_type_ids']]

        return (inp,tgt,lbl)

    def __len__(self):
        return self.size

class HypothesisOnlyDataset(Dataset):

    def __init__(self, lines, labels, tokenizer, sep='|||', max_len=512):
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = _labels_to_idx(labels)
        self.tokenizer = tokenizer
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)

    @staticmethod    
    def _labels_to_idx(labels):
        labels_ids = list(set(labels))
        res = [labels_ids.index(label) for label in labels]
        
        return res

    def __getitem__(self, index):

        split = self.lines[index].split(self.sep)

        inp = split[0]
        # tgt = split[1].replace('\n', '')
        lbl = torch.tensor(self.labels[index])

        # input_dict = self.tokenizer.encode_plus(inp,
        #                                         max_length=self.max_len,
        #                                         pad_to_max_length=True,
        #                                         return_tensors='pt',
        #                                         )

        # res = [input_dict[item].squeeze(0) for item in ['input_ids', 'attention_mask']]

        return (inp,lbl)

    def __len__(self):
        return self.size
