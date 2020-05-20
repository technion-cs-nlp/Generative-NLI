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

        input_dict = self.tokenizer_encoder.encode_plus(sent,
                                                max_length=self.max_len,
                                                pad_to_max_length=True
                                                )
        target_dict = self.tokenizer_decoder.encode_plus(tgt,
                                                max_length=self.max_len,
                                                pad_to_max_length=True
                                                )

        decoder_encoder_attention_mask = self.tokenizer_decoder.encode_plus(sent,
                                                max_length=self.max_len,
                                                pad_to_max_length=True
                                                )['attention_mask']
        # except Exception as e:
        #     print(e)
        #     print(tgt)
        #     import pdb; pdb.set_trace()

        res = [torch.tensor(input_dict[item]) for item in ['input_ids', 'attention_mask']] + \
              [torch.tensor(target_dict[item]) for item in ['input_ids', 'attention_mask']]

        # res = [torch.tensor(input_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']] + \
        #       [torch.tensor(target_dict[item]) for item in ['input_ids', 'attention_mask', 'token_type_ids']]

        return tuple(res) + (torch.tensor(decoder_encoder_attention_mask),)

    def __len__(self):
        return self.size
