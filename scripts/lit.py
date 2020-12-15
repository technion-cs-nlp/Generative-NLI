import lit_nlp
import lit_nlp.api.types as lit_types
from src.models import get_model
from lit_nlp.api import model as lit_model
from lit_nlp.api import dataset as lit_dataset
from lit_nlp import dev_server

from src.train import GenerativeTrainer
from transformers import AutoTokenizer, AdamW, AutoModel, \
    get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, \
    get_constant_schedule_with_warmup
from src.models import get_model, HybridModel
from src.train import GenerativeTrainer, DiscriminativeTrainer, OnelabelTrainer, HybridTrainer
from src.data import PremiseGenerationDataset, DiscriminativeDataset, HypothesisOnlyDataset, DualDataset
import torch
import os
from absl import flags
from absl import app
import pathlib
from lit_nlp import server_flags


FLAGS = flags.FLAGS
# FLAGS.set_default(
#     "client_root",
#     os.path.join(pathlib.Path(__file__).parent.absolute(), "build"))


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


class NLIModel(lit_model.Model):
    """Wrapper for a Natural Language Inference model."""

    NLI_LABELS = ['entailment', 'neutral', 'contradiction']
    NLI_LABELS.sort()

    def __init__(self, model_path, **kw):
        # Load the model into memory so we're ready for interactive use.
        self.model_name = "sshleifer/distilbart-cnn-12-6"
        self.model_path = model_path
        self._model, self._trainer = self.create_trainer(model_path=self.model_path, model_name=self.model_name,
                                                         NLI_labels=NLIModel.NLI_LABELS)

    ##
    # LIT API implementations
    def max_minibatch_size(self):
        # This tells lit_model.Model.predict() how to batch inputs to
        # predict_minibatch().
        # Alternately, you can just override predict() and handle batching yourself.
        return 8
    
    def predict_minibatch(self, inputs, config=None):
        """Predict on a single minibatch of examples.
        """
        examples = (inputs['premise'], inputs['hypothesis'], None)  # any custom preprocessing
        return self._trainer.test_batch(examples)  # returns a dict for each input

    def input_spec(self):
        """Describe the inputs to the model."""
        return {
            'premise': lit_types.TextSegment(),
            'hypothesis': lit_types.TextSegment(),
        }

    def output_spec(self):
        """Describe the model outputs."""
        return {
            # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
            'probas': lit_types.MulticlassPreds(vocab=NLIModel.NLI_LABELS, parent='label'),
        }

    def create_trainer(self, model_path, model_name, NLI_labels, decoder_model_name=None, model_type='bart',
                       label=None, save_results=False, create_premises=False):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 'gpt' in model_name:
            tokenizer.pad_token = tokenizer.unk_token
        tokenizer_decoder = None
        if decoder_model_name is not None and 'gpt' in decoder_model_name:
            from transformers import GPT2Tokenizer
            GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
            tokenizer_decoder = GPT2Tokenizer.from_pretrained(decoder_model_name)
            tokenizer_decoder.pad_token = tokenizer_decoder.unk_token

        all_labels_text = NLI_labels
        all_labels_text.sort()
        num_labels = len(all_labels_text)

        if not model_type.startswith('disc'):
            all_labels = ['[' + l.upper().replace('\n', '') + ']' for l in all_labels_text]

            tokenizer.add_tokens(all_labels)
            labels_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in all_labels]
            print(f'Labels IDs: {labels_ids}')

        dataset = None
        trainer_type = None
        data_args = {}
        dataloader_args = {}
        train_args = {}

        train_args['save_results'] = save_results
        if model_type in ['encode-decode', 'bart', 'shared']:
            dataset = DiscriminativeDataset
            if label is None:
                trainer_type = GenerativeTrainer
            else:
                trainer_type = OnelabelTrainer
            # data_args['tokenizer_decoder'] = tokenizer_decoder
            # data_args['generate_hypothesis'] = generate_hypothesis
            train_args['possible_labels_ids'] = labels_ids
            train_args['tokenizer_encoder'] = tokenizer
            train_args['tokenizer_decoder'] = tokenizer_decoder
            train_args['create_premises'] = create_premises
            # dataloader_args['collate_fn'] = my_collate
        elif model_type.startswith('disc'):
            trainer_type = DiscriminativeTrainer
            train_args['num_labels'] = num_labels
            train_args['tokenizer'] = tokenizer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = get_model(tokenizer=tokenizer, tokenizer_decoder=tokenizer_decoder, model=model_type,
                          model_name=model_name,
                          decoder_model_name=decoder_model_name, model_path=model_path)

        trainer = trainer_type(model, optimizer=None, scheduler=None, device=device, **train_args)

        return model, trainer


def main(_):
    # MulitiNLIData implements the Dataset API
    datasets = {
        'snli': SNLIData('./data/snli_1.0/cl_snli'),
    }

    # NLIModel implements the Model API
    models = {
        'model': NLIModel("checkpoints/bart_mini_ft_model"),
    }

    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    lit_demo.serve()


if __name__ == '__main__':
    app.run(main)
