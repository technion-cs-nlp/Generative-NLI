import torch.nn as nn
import os
import torch

class HybridModel(nn.Module):
    def __init__(self, model1, model2, gamma):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, args1, args2,**kwargs):
        out1 = self.model1(**kwargs)
        loss1 = out1[0]
        out2 = self.model2(**kwargs)
        loss2 = out2[0]
        res = gamma * loss1 + (1-gamma) * loss2
        return res


def freeze_params(params, ratio):
    for idx, param in enumerate(params):
        if idx >= len(params) * ratio:
            break
        param.requires_grad = False

def get_model(model='encode-decode', model_name='bert-base-uncased', tokenizer=None,
            tokenizer_decoder=None, decoder_model_name=None, 
            model_path=None, param_freezing_ratio=0.0, num_labels=3, tie_embeddings=False,
            label=None, gamma=0.5):
    res_model = None
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_list = ['encode-decode', 'decoder-only', 'bart', 'discriminative', 'shared', 'hybrid']
    if model == 'encode-decode':
        from transformers import EncoderDecoderModel, AutoConfig, EncoderDecoderConfig
        if decoder_model_name is None:
            decoder_model_name = model_name

        encoder_model_name, decoder_model_name = (model_name, decoder_model_name)

        if model_path is None:
            res_model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_name, decoder_model_name, tie_encoder_decoder=tie_embeddings)
            if label is None:
                res_model.encoder.resize_token_embeddings(len(tokenizer))

            # if tie_embeddings:
            #     res_model.decoder.cls.predictions.decoder.weight.data = res_model.encoder.embeddings.word_embeddings.weight.data
            #     res_model.decoder.bert.embeddings.word_embeddings.weight.data = res_model.encoder.embeddings.word_embeddings.weight.data

        else:
            res_model = EncoderDecoderModel.from_pretrained(model_path)

        if tokenizer_decoder is None:
            res_model.config.decoder_start_token_id = tokenizer.cls_token_id
            res_model.config.eos_token_id = tokenizer.sep_token_id
            res_model.config.pad_token_id = tokenizer.pad_token_id
        else:
            res_model.config.decoder_start_token_id = tokenizer_decoder.bos_token_id
            res_model.config.eos_token_id = tokenizer_decoder.eos_token_id
        res_model.config.max_length = 120
        res_model.config.min_length = 5
        res_model.config.no_repeat_ngram_size = 3
        res_model.early_stopping = True
        res_model.length_penalty = 2.0
        res_model.num_beams = 4

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

    elif model == 'hybrid':
        from transformers import BartForConditionalGeneration, BartForSequenceClassification
        model1 = BartForConditionalGeneration.from_pretrained(model_name)
        model2 = BartForSequenceClassification.from_pretrained(model_name)
        del model2.model
        model2.model = model1.model
        model1.resize_token_embeddings(len(tokenizer))
        res_model = HybridModel(model1,model2,gamma)
        return res_model

    else: 
        print(f"Please pick a valid model in {model_list}")

    if model_path is None and not 'discriminative'.startswith(model) \
        and label is None:          ## only change embeddings size if its not a trained model
        res_model.resize_token_embeddings(len(tokenizer))
        
    return res_model
