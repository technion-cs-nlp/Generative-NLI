import torch.nn as nn
import os
from transformers import BertForMaskedLM, PreTrainedEncoderDecoder, BertModel, \
    GPT2LMHeadModel, AutoConfig, AutoTokenizer, BartForConditionalGeneration


class PremiseGeneratorHybrid(nn.Module):
    def __init__(self, bert_name, gpt_name, tokenizers):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_name)
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt_name, is_decoder=True)
        self.bert_tokenizer = tokenizers[0]
        self.gpt_tokenizer = tokenizers[1]

        self.encoder.resize_token_embeddings(len(self.bert_tokenizer))
        self.decoder.resize_token_embeddings(len(self.gpt_tokenizer))

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
        kwargs_decoder["labels"] = kwargs_decoder.pop("lm_labels", None)
        kwargs_decoder.pop("encoder_attention_mask", None)
        decoder_outputs = self.decoder(decoder_input_ids, **kwargs_decoder)

        return decoder_outputs + encoder_outputs

def freeze_params(params, ratio):
    for idx, param in enumerate(params):
        if idx > len(params) * ratio:
            break
        param.requires_grad = False

def get_model(model='encode-decode', model_name='bert-base-uncased', tokenizer=None, model_name_decoder=None, 
            model_path=None):
    res_model = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_list = ['masked', 'encode-decode', 'hybrid', 'decoder', 'bart']

    if model == 'encode-decode':
        encoder_model_name, decoder_model_name = \
                        (os.path.join(model_path,'encoder'), os.path.join(model_path,'decoder')) \
                        if model_path is not None else (model_name, model_name)

        decoder_config = AutoConfig.from_pretrained(decoder_model_name, is_decoder=True)
        res_model = PreTrainedEncoderDecoder.from_pretrained(encoder_model_name, decoder_model_name,
                                                                decoder_config=decoder_config)

        res_model.encoder.resize_token_embeddings(len(tokenizer))
        res_model.decoder.resize_token_embeddings(len(tokenizer))

        params_enc = list(res_model.encoder.parameters())
        params_dec = list(res_model.decoder.parameters())

        param_freezing_ratio = 1
        freeze_params(params_enc, param_freezing_ratio)
        freeze_params(params_dec, param_freezing_ratio)

        return res_model
    
    else:
        model_name = model_path if model_path is not None else model_name

    if model  == 'bart':
        res_model = BartForConditionalGeneration.from_pretrained(model_name)

    if model == 'masked':
        res_model = BertForMaskedLM.from_pretrained(model_name)

    elif model == 'hybrid':
        decoder_tokenizer = AutoTokenizer.from_pretrained(model_name_decoder)
        res_model = PremiseGeneratorHybrid(model_name, model_name_decoder, [tokenizer, decoder_tokenizer])

    elif model == 'decoder':
        res_model = GPT2LMHeadModel.from_pretrained(model_name)

    else: 
        print(f"Please pick a valid model in {model_list}")

    res_model.resize_token_embeddings(len(tokenizer))
        
    return res_model
