import torch.nn as nn
import os
import torch


def freeze_params(params, ratio):
    for idx, param in enumerate(params):
        if idx >= len(params) * ratio:
            break
        param.requires_grad = False

def get_model(model='encode-decode', model_name='bert-base-uncased', tokenizer=None,
            tokenizer_decoder=None, decoder_model_name=None, 
            model_path=None, param_freezing_ratio=0.0, num_labels=3, tie_embeddings=False,
            label=None, gamma=0.5, tie_encoder_decoder=False):
    res_model = None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_list = ['encode-decode', 'decoder-only', 'bart', 'discriminative', 'shared', 'hybrid', 't5']
    if model == 'encode-decode':
        from transformers import EncoderDecoderModel, AutoConfig, EncoderDecoderConfig
        if decoder_model_name is None:
            decoder_model_name = model_name

        encoder_model_name, decoder_model_name = (model_name, decoder_model_name)

        if model_path is None:
            encoder_decoder_params = {
                'tie_encoder_decoder':tie_embeddings,
            }
            res_model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_name, decoder_model_name, **encoder_decoder_params)
            if "bert" in encoder_model_name:
                res_model.config.decoder_start_token_id = tokenizer.cls_token_id
                res_model.config.eos_token_id = tokenizer.sep_token_id
                res_model.config.pad_token_id = tokenizer.pad_token_id
            if label is None:
                res_model.encoder.resize_token_embeddings(len(tokenizer))
                res_model.config.encoder.vocab_size = len(tokenizer)
            res_model.config.vocab_size = res_model.config.encoder.vocab_size

        else:
            res_model = EncoderDecoderModel.from_pretrained(model_path)
            if 'patrick' in model_path:
                pass
                res_model.encoder.resize_token_embeddings(len(tokenizer))
                res_model.config.encoder.vocab_size = len(tokenizer)
                res_model.config.vocab_size = len(tokenizer)

        res_model.config.max_length = 64
        res_model.config.min_length = 5
        res_model.config.no_repeat_ngram_size = 3
        res_model.early_stopping = True
        res_model.length_penalty = 2.0
        res_model.num_beams = 4

        return res_model
    
    else:
        name = model_name
        model_name = model_path if model_path is not None else model_name
    if model == 'bert2bert':
        if model_path is None:
            from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
            encoder = BertGenerationEncoder.from_pretrained(model_name, bos_token_id=101, eos_token_id=102)
            # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
            decoder = BertGenerationDecoder.from_pretrained(model_name, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
            bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder, tie_encoder_decoder=tie_encoder_decoder)
            if label is None:
                    bert2bert.encoder.resize_token_embeddings(len(tokenizer))
                    bert2bert.config.encoder.vocab_size = len(tokenizer)
        else:
            from transformers import EncoderDecoderModel
            bert2bert = EncoderDecoderModel.from_pretrained(model_path)
        return bert2bert
    elif model  == 'bart':
        from transformers import BartForConditionalGeneration
        res_model = BartForConditionalGeneration.from_pretrained(model_name)
    
    elif model == 't5':
        from transformers import T5ForConditionalGeneration
        res_model = T5ForConditionalGeneration.from_pretrained(model_name)

    elif model == 'decoder-only':
        from transformers import AutoModelForCausalLM
        args = {}
        if 'gpt' not in model_name:
            args['is_decoder']=True
        res_model = AutoModelForCausalLM.from_pretrained(model_name, **args)
        if 'gpt' in model_name:
            res_model.config.pad_token_id = tokenizer.pad_token_id

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
        else:
            from transformers import EncoderDecoderModel
            res_model = EncoderDecoderModel.from_pretrained(model_path)

        return res_model

    elif 'discriminative'.startswith(model):
        if 't5' in name:
            from transformers import T5ForConditionalGeneration
            res_model = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            from transformers import AutoModelForSequenceClassification
            if model_path is None:
                res_model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels, return_dict=False)
                if 'gpt' in model_name:
                    res_model.config.pad_token_id = tokenizer.pad_token_id
            else:
                res_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    else: 
        print(f"Please pick a valid model in {model_list}")

    if model_path is None and not 'discriminative'.startswith(model) \
        and label is None:          ## only change embeddings size if its not a trained model
        # pass
        res_model.resize_token_embeddings(len(tokenizer))
        if hasattr(res_model.config,"encoder"):
            res_model.config.encoder.vocab_size = len(tokenizer)
        else:
            res_model.config.vocab_size = len(tokenizer)
    return res_model
