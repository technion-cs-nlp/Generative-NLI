from lit_nlp import dev_server
from lit_nlp.api import model as lit_model
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
from lit_nlp import server_flags

import torch
from torch.autograd import grad
import numpy as np
from copy import deepcopy
from src.data import PremiseGenerationDataset, DiscriminativeDataset
from transformers import BartForSequenceClassification, AutoTokenizer,BertForSequenceClassification

from absl import flags
from absl import app


class NLIDatasetWrapper(lit_dataset.Dataset):
    """Loader for MultiNLI development set."""

    NLI_LABELS = ['entailment', 'neutral', 'contradiction']
    NLI_LABELS.sort()

    def __init__(self, ds, path):
        # Store as a list of dicts, conforming to self.spec()
        if ds is None:
            test_lines, test_labels = self.read_file(path)
            ds = DiscriminativeDataset(test_lines, test_labels)
        examples = []
        for ind in range(len(ds)):
            p, h, y = ds[ind]
            examples.append({'premise': p, 'hypothesis': h, 'label': self.NLI_LABELS[y], 'grad_class':self.NLI_LABELS[y]})
        self._examples = examples

    def spec(self):
        return {
            'premise': lit_types.TextSegment(),
            'hypothesis': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=self.NLI_LABELS),
            'grad_class': lit_types.CategoryLabel(vocab=self.NLI_LABELS),
        }

    @staticmethod
    def read_file(data_dir_prefix):
        test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')
        with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
            test_labels = test_labels_file.readlines()
        with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
            test_lines = test_lines_file.readlines()

        return test_lines, test_labels

class SNLIData(lit_dataset.Dataset):
    NLI_LABELS = ['entailment', 'neutral', 'contradiction']
    NLI_LABELS.sort()

    def __init__(self, path):
        # Read the eval set from a .tsv file as distributed with the GLUE benchmark.
        test_lines, test_labels = self.read_file(path)
        # Store as a list of dicts, conforming to self.spec()
        self._examples = [{
            'premise': test_line.split('|||')[0].strip(),
            'hypothesis': test_line.split('|||')[1].strip(),
            'label': test_label.strip(),
        } for test_line, test_label in zip(test_lines, test_labels)]

    def spec(self):
        return {
            'premise': lit_types.TextSegment(),
            'hypothesis': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=self.NLI_LABELS),
        }

    @staticmethod
    def read_file(data_dir_prefix):
        test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')
        with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
            test_labels = test_labels_file.readlines()
        with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
            test_lines = test_lines_file.readlines()

        return test_lines, test_labels


class NLIModelWrapper(lit_model.Model):
    LABELS = ['entailment', 'neutral', 'contradiction']
    LABELS.sort()

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

    # LIT API implementation
    def max_minibatch_size(self):
        # This tells lit_model.Model.predict() how to batch inputs to
        # predict_minibatch().
        # Alternately, you can just override predict() and handle batching yourself.
        return 8

    def predict_minibatch(self, inputs):
        p, h = list(map(lambda x: x['premise'], inputs)), list(map(lambda x: x['hypothesis'], inputs))
        input_dict = self.tokenizer(p, h, padding=True, truncation=True, return_tensors='pt')  # tensors - shape B x S
        batched_outputs = {}

        # Check and send to cuda (GPU) if available
        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in input_dict:
                input_dict[tensor] = input_dict[tensor].cuda()

        # for integrated gradients - precalculate word embeddings and pass them in instead of input_ids to enable
        # .grad of output with respect to embeddings
        input_ids = input_dict["input_ids"]
        word_embeddings = self.model.bert.embeddings.word_embeddings
        input_embs = word_embeddings(input_ids)  # tensor of shape B x S x h
        input_embs = scatter_embs(input_embs, inputs)
        model_inputs = input_dict.copy()
        model_inputs["input_ids"] = None

        logits, hidden_states, unused_attentions = self.model(**model_inputs, inputs_embeds=input_embs)

        # for integrated gradients - Choose output to "explain"(from num_labels) according to grad_class
        grad_classes = [self.LABELS.index(ex["grad_class"]) for ex in inputs]  # list of length B of integer indices
        indices = np.arange(len(grad_classes)).tolist(), grad_classes
        scalar_pred_for_gradients = logits[indices]

        # prepare model outputs according to output_spec
        # all values must be numpy arrays or tensor (have the shape attribute) in order to unbatch them
        batched_outputs["input_emb_grads"] = grad(scalar_pred_for_gradients, input_embs,
                                                  torch.ones_like(scalar_pred_for_gradients))[0].detach().numpy()

        batched_outputs["probas"] = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()  # B x num_labels
        batched_outputs["input_ids"] = input_dict["input_ids"].detach().numpy()  # B x S
        batched_outputs["cls_pooled"] = hidden_states[-1][:, 0, :].detach().numpy()  # output of embeddings layer, B x h
        batched_outputs["input_embs"] = input_embs.detach().numpy()  # B x S x h
        batched_outputs["grad_class"] = np.array([ex["grad_class"] for ex in inputs])

        # Unbatch outputs so we get one record per input example.
        for output in utils.unbatch_preds(batched_outputs):
            output["tokens"] = self.tokenizer.convert_ids_to_tokens(output.pop("input_ids"))  # list of length seq
            output = self._postprocess(output)
            yield output

    def _postprocess(self, output_samp):
        special_tokens_mask = list(map(lambda x: x != self.tokenizer.pad_token, output_samp['tokens']))
        output_samp['tokens'] = (np.array(output_samp['tokens'])[special_tokens_mask]).tolist()
        output_samp['input_embs'] = output_samp['input_embs'][special_tokens_mask]
        output_samp['input_emb_grads'] = output_samp['input_emb_grads'][special_tokens_mask]
        return output_samp

    def input_spec(self) -> lit_types.Spec:
        inputs = {}
        inputs["premise"] = lit_types.TextSegment()
        inputs["hypothesis"] = lit_types.TextSegment()

        # for gradient attribution
        inputs["input_embs"] = lit_types.TokenEmbeddings(required=False)
        inputs["grad_class"] = lit_types.CategoryLabel(vocab=self.LABELS)

        return inputs

    def output_spec(self) -> lit_types.Spec:
        output = {}
        output["tokens"] = lit_types.Tokens()
        output["probas"] = lit_types.MulticlassPreds(parent="label", vocab=self.LABELS)
        output["cls_pooled"] = lit_types.Embeddings()

        # for gradient attribution
        output["input_embs"] = lit_types.TokenEmbeddings()
        output["grad_class"] = lit_types.CategoryLabel(vocab=self.LABELS)
        output["input_emb_grads"] = lit_types.TokenGradients(align="tokens",
                                                             grad_for="input_embs", grad_target="grad_class")

        return output


def scatter_embs(input_embs, inputs):
    """
    For inputs that have 'input_embs' field passed in, replace the entry in input_embs[i] with the entry
    from inputs[i]['input_embs']. This is useful for the Integrated Gradients - for which the predict is
    called with inputs with 'input_embs' field which is an interpolation between the baseline and the real calculated
    input embeddings for the sample.
    :param input_embs: tensor of shape B x S x h of input embeddings according to the input sentences.
    :param inputs: list of dictionaries (smaples), for which the 'input_embs' field might be specified
    :return: tensor of shape B x S x h with embeddings (if passed) from inputs inserted to input_embs
    """
    interp_embeds = [(ind, ex.get('input_embs')) for ind, ex in enumerate(inputs)]
    for ind, embed in interp_embeds:
        if embed is not None:
            input_embs[ind] = torch.tensor(embed)

    return input_embs

def main(_):
    # MulitiNLIData implements the Dataset API
    datasets = {
        'snli': NLIDatasetWrapper(ds=None,path='./data/snli_1.0/cl_snli'),
    }
#     model_name = "checkpoints/bart_mini_disc_model/_model"
    model_name = "checkpoints/bert_disc_model"
#     tokenizer_name = 'sshleifer/distilbart-cnn-12-6'
    tokenizer_name = 'bert-base-uncased'

    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=3,
                                              output_hidden_states=True,
                                              output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    all_labels_text = datasets['snli'].NLI_LABELS
    all_labels_text.sort()
    num_labels = len(all_labels_text)
    all_labels = ['[' + l.upper().replace('\n', '') + ']' for l in all_labels_text]
    tokenizer.add_tokens(all_labels)
    labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]
    print(f'Labels IDs: {labels_ids}')

    models = {
        'model': NLIModelWrapper(tokenizer,model),
    }
    flags = server_flags.get_flags()
    flags['port'] = 16006

    lit_demo = dev_server.Server(models, datasets, **flags)
    lit_demo.serve()


if __name__ == '__main__':
    app.run(main)
