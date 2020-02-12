import torch
from torch.utils.data import Dataset


class PremiseGenerationDataset(Dataset):

    def __init__(self, lines, labels, tokenizer, sep='|||', max_len=512):
        assert len(lines) == len(labels)
        super().__init__()
        self.lines = lines
        self.labels = labels
        self.tokenizer = tokenizer
        self.sep = sep
        self.max_len = max_len
        self.size = len(self.lines)

    def __getitem__(self, index):
        split = self.lines[index].split(self.sep)

        inp = split[0]
        tgt = split[1]

        sent = '<' + self.labels[index][:-1] + '> ' + inp

        input_dict = self.tokenizer.encode_plus(sent,
                                                max_length=self.max_len,
                                                pad_to_max_length=True)
        target_dict = self.tokenizer.encode_plus(tgt,
                                                 max_length=self.max_len,
                                                 pad_to_max_length=True)

        res = [torch.tensor(input_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']] + \
              [torch.tensor(target_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']]

        return tuple(res)

    def __len__(self):
        return self.size
