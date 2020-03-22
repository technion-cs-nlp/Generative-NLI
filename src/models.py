import torch.nn as nn
from transformers import BertForMaskedLM, PreTrainedEncoderDecoder, BertModel, \
    GPT2LMHeadModel, AutoConfig, AutoTokenizer


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


def get_model(model='encode-decode', model_name='bert-base-uncased', tokenizer=None, model_name_decoder=None):
    res_model = None
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_list = ['masked', 'encode-decode', 'hybrid', 'decoder']

    if model == 'encode-decode':
        decoder_config = AutoConfig.from_pretrained(model_name, is_decoder=True)
        res_model = PreTrainedEncoderDecoder.from_pretrained(model_name, model_name,
                                                                decoder_config=decoder_config)

        res_model.encoder.resize_token_embeddings(len(tokenizer))
        res_model.decoder.resize_token_embeddings(len(tokenizer))

    elif model == 'masked':
        res_model = BertForMaskedLM.from_pretrained(model_name)
        res_model.resize_token_embeddings(len(tokenizer))

    elif model == 'hybrid':
        decoder_tokenizer = AutoTokenizer.from_pretrained(model_name_decoder)
        res_model = PremiseGeneratorHybrid(model_name, model_name_decoder, [tokenizer, decoder_tokenizer])

    elif model == 'decoder':
        res_model = GPT2LMHeadModel.from_pretrained(model_name)
        res_model.resize_token_embeddings(len(tokenizer))

    else: 
        print(f"Please pick a valid model in {model_list}")
        
    return res_model
