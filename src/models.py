import torch
import torch.nn as nn
import transformers
from transformers import Model2Model


class PremiseGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = Model2Model.from_pretrained(model_name)

    def forward(self, encoder_input_ids, decoder_input_ids, train=False, **kwargs):
        if train:
            self.model.train()
        else:
            self.model.eval()

        res = self.model(encoder_input_ids, decoder_input_ids, **kwargs)

        return res


