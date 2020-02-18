import torch
import torch.nn as nn
import transformers
from transformers import Model2Model, BertTokenizer, BertForMaskedLM


class PremiseGenerator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = Model2Model.from_pretrained(model_name)

    def forward(self, encoder_input_ids, decoder_input_ids, train=False, **kwargs):
        if train:
            self.model.train()
        else:
            self.model.eval()

        res = self.model(encoder_input_ids, decoder_input_ids, **kwargs)

        return res


class PremiseGeneratorMaskedLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()


def get_model(model='masked', model_name='bert-base-uncased', tokenizer=None):
    res_model = None
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    assert model in ['masked', 'encode-decode']

    if model == 'encode-decode':
        res_model = Model2Model.from_pretrained(model_name)
        res_model.encoder.resize_token_embeddings(len(tokenizer))
        res_model.decoder.resize_token_embeddings(len(tokenizer))

    elif model == 'masked':
        res_model = BertForMaskedLM.from_pretrained(model_name)
        res_model.resize_token_embeddings(len(tokenizer))

    return res_model




