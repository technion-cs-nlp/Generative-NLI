import torch.nn as nn
import os

def freeze_params(params, ratio):
    for idx, param in enumerate(params):
        if idx >= len(params) * ratio:
            break
        param.requires_grad = False

def get_model(model='encode-decode', model_name='bert-base-uncased', tokenizer=None,
            tokenizer_decoder=None, decoder_model_name=None, 
            model_path=None, param_freezing_ratio=0.0, num_labels=3):
    res_model = None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_list = ['masked', 'encode-decode', 'hybrid', 'decoder', 'bart', 'discriminitive']
    if model == 'encode-decode':
        from transformers import EncoderDecoderModel, AutoConfig, EncoderDecoderConfig
        if decoder_model_name is None:
            decoder_model_name = model_name
        # encoder_model_name, decoder_model_name = \
        #                 (os.path.join(model_path,'encoder'), os.path.join(model_path,'decoder')) \
        #                 if model_path is not None else (model_name, decoder_model_name)

        encoder_model_name, decoder_model_name = (model_name, decoder_model_name)

        if tokenizer_decoder is None:
            tokenizer_decoder = tokenizer

        if model_path is None:
            res_model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_name, decoder_model_name)
            res_model.encoder.resize_token_embeddings(len(tokenizer))
            res_model.decoder.resize_token_embeddings(len(tokenizer_decoder))
        else:
            res_model = EncoderDecoderModel.from_pretrained(model_path)

        return res_model
    
    else:
        model_name = model_path if model_path is not None else model_name

    if model  == 'bart':
        from transformers import BartForConditionalGeneration
        res_model = BartForConditionalGeneration.from_pretrained(model_name)

    elif model == 'masked':
        from transformers import BertForMaskedLM
        res_model = BertForMaskedLM.from_pretrained(model_name)

    elif model == 'decoder':
        from transformers import GPT2LMHeadModel
        res_model = GPT2LMHeadModel.from_pretrained(model_name)

    elif 'discriminitive'.startswith(model):
        from transformers import AutoModelForSequenceClassification
        res_model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)

    else: 
        print(f"Please pick a valid model in {model_list}")

    if model_path is None and not 'discriminitive'.startswith(model):          ## only change embeddings size if its not a trained model
        # import pdb; pdb.set_trace()
        res_model.resize_token_embeddings(len(tokenizer))
        
    return res_model
