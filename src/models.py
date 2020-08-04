import torch.nn as nn
import os

def freeze_params(params, ratio):
    for idx, param in enumerate(params):
        if idx >= len(params) * ratio:
            break
        param.requires_grad = False

def get_model(model='encode-decode', model_name='bert-base-uncased', tokenizer=None,
            tokenizer_decoder=None, decoder_model_name=None, 
            model_path=None, param_freezing_ratio=0.0, num_labels=3, tie_embeddings=False):
    res_model = None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_list = ['encode-decode', 'decoder-only', 'bart', 'discriminative', 'shared']
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

            if tie_embeddings:
                res_model.decoder.cls.predictions.decoder.weight.data = res_model.encoder.embeddings.word_embeddings.weight.data
                res_model.decoder.bert.embeddings.word_embeddings.weight.data = res_model.encoder.embeddings.word_embeddings.weight.data
        else:
            # import pdb; pdb.set_trace()
            res_model = EncoderDecoderModel.from_pretrained(model_path)

        return res_model
    
    else:
        model_name = model_path if model_path is not None else model_name

    if model  == 'bart':
        from transformers import BartForConditionalGeneration
        res_model = BartForConditionalGeneration.from_pretrained(model_name)

    elif model == 'decoder-only':
        from transformers import AutoModelForCausalLM
        res_model = AutoModelForCausalLM.from_pretrained(model_name)

    elif model == 'shared':
        if model_path is None:
            from transformers import AutoModelForCausalLM, EncoderDecoderModel
            decoder = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)
            try:
                name = model_name.split('-')[0]
                encoder = getattr(decoder, name)
            except Exception as e:
                raise AttributeError(f"Can't use share model with {model_name} architecture")

            res_model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
            res_model.encoder.resize_token_embeddings(len(tokenizer))
            res_model.decoder.resize_token_embeddings(len(tokenizer))
        else:
            from transformers import EncoderDecoderModel
            res_model = EncoderDecoderModel.from_pretrained(model_path)

        return res_model

    elif 'discriminative'.startswith(model):
        from transformers import AutoModelForSequenceClassification
        if model_path is None:
            res_model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
        else:
            # import pdb; pdb.set_trace()
            res_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    else: 
        print(f"Please pick a valid model in {model_list}")

    if model_path is None and not 'discriminative'.startswith(model):          ## only change embeddings size if its not a trained model
        # import pdb; pdb.set_trace()
        res_model.resize_token_embeddings(len(tokenizer))
        
    return res_model
