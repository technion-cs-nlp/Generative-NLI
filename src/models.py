import torch
import torch.nn as nn
import transformers
from transformers import Model2Model, BertTokenizer, BertForMaskedLM, PreTrainedEncoderDecoder, BertModel, \
    GPT2Tokenizer, GPT2LMHeadModel


class PremiseGeneratorHybrid(nn.Module):
    def __init__(self, bert_name, gpt_name, bert_tokenizer, gpt_tokenizer):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_name)
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt_name, is_decoder=True)

        self.encoder.resize_token_embeddings(len(bert_tokenizer))
        self.decoder.resize_token_embeddings(len(gpt_tokenizer))

    def forward(self, encoder_input_ids, decoder_input_ids, **kwargs):
        kwargs_encoder, kwargs_decoder = Model2Model.prepare_model_kwargs(
            **kwargs)  # Need to check if this is compatible

        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_outputs = ()

        kwargs_decoder["inputs_embeds"] = encoder_hidden_states
        decoder_outputs = self.decoder(decoder_input_ids, **kwargs_decoder)

        return decoder_outputs + encoder_outputs


def get_model(model='masked', model_name='bert-base-uncased', tokenizer=None, v=2):
    res_model = None
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    assert model in ['masked', 'encode-decode'], f"Please pick a valid model in {['masked', 'encode-decode']}"

    if model == 'encode-decode':
        if v == 1:
            res_model = PreTrainedEncoderDecoder(
                BertModel.from_pretrained(model_name, output_hidden_states=True,
                                          # output_attentions=True
                                          ),
                BertForMaskedLM.from_pretrained(model_name, is_decoder=True)
            )
        if v == 2:
            res_model = Model2Model.from_pretrained(model_name)

        res_model.encoder.resize_token_embeddings(len(tokenizer))
        res_model.decoder.resize_token_embeddings(len(tokenizer))

    elif model == 'masked':
        res_model = BertForMaskedLM.from_pretrained(model_name)
        res_model.resize_token_embeddings(len(tokenizer))

    return res_model
