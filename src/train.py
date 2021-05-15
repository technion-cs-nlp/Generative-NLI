import abc
import os
import pdb
import sys
import numpy as np
from torch._C import device
from torch.serialization import validate_cuda_device
import tqdm
import torch
import nlp

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Any
from pathlib import Path
from src.utils import BatchResult, EpochResult, FitResult

from src.loss import LabelSmoothingCrossEntropy

from transformers import GPT2Tokenizer

temp =[0,0,0]
return_acc = False
mode = 'test'

def gen_loss(i, ret_val, model, model_kwargs, inp_x, attention, reduce, reduction):
                outputs = model(**model_kwargs)
                loss = outputs[0]
                loss = loss.view(inp_x.size(0), -1)
                attention = attention[:, 1:] if loss.shape != attention.shape else attention
                # import pdb; pdb.set_trace()
                loss = reduce(loss, attention=attention, reduction=reduction)
                ret_val.value = loss
            
def hyp_loss(i,ret_val, num_labels, batch_size, batch, calc_disc_loss):
    test_labels = torch.arange(num_labels).repeat(batch_size,1).T.reshape(-1)  # (0,...,0,1,...,1,2,...,2)
    hyp_batch_test = (None, batch[1]*3, test_labels)
    prior = calc_disc_loss(hyp_batch_test)
    ret_val.value = prior


def create_args_generative(batch, labels, device):
    x, encoder_attention_mask, y, decoder_attention_mask = batch

    label_loc = 1

    batch_size = x.size(0)

    inp_x = []
    inp_y = []
    inp_e_a_m = []
    inp_d_a_m = []
    # inp_d_e_a_m = []

    for label_id in labels:
        curr_x = x.clone()
        curr_x[:, label_loc] = label_id
        inp_x.append(curr_x)
        inp_y.append(y)
        inp_e_a_m.append(encoder_attention_mask)
        inp_d_a_m.append(decoder_attention_mask)
        # inp_d_e_a_m.append(decoder_encoder_attention_mask)

    inp_x = torch.cat(inp_x)
    inp_y = torch.cat(inp_y)
    inp_e_a_m = torch.cat(inp_e_a_m)
    inp_d_a_m = torch.cat(inp_d_a_m)
    # inp_d_e_a_m = torch.cat(inp_d_e_a_m)

    inp_x = inp_x.to(device)
    inp_y = inp_y.to(device)
    inp_e_a_m = inp_e_a_m.to(device)
    inp_d_a_m = inp_d_a_m.to(device)
    # inp_d_e_a_m = inp_d_e_a_m.to(self.device)

    model_kwargs = {
        "input_ids": inp_x,
        "decoder_input_ids": inp_y,
        "attention_mask": inp_e_a_m,
        "decoder_attention_mask": inp_d_a_m,
        "labels": inp_y
    }

    return model_kwargs


def create_args_discriminitive(batch, tokenizer, device):
    if len(batch) == 3:  # H, P, y
        H, P, labels = batch
        input_dict = tokenizer.batch_encode_plus([[H[i], P[i]] for i in range(len(H))], padding='longest',
                                                 return_tensors='pt', truncation=True)
    elif len(batch) == 2:  # Hypotesis only
        H, labels = batch
        input_dict = tokenizer.batch_encode_plus(H, padding='longest', return_tensors='pt', truncation=True)
    else:
        labels = None
        input_dict = None
    batch_encoded = [input_dict[item] for item in ['input_ids', 'attention_mask']]
    x, attention_mask = batch_encoded
    x = x.to(device)
    attention_mask = attention_mask.to(device)
    # token_type_ids = token_type_ids.to(self.device)
    labels = labels.to(device)

    model_kwargs = {
        "input_ids": x,
        "attention_mask": attention_mask,
        # "token_type_ids": token_type_ids,
        "labels": labels
    }

    if 'token_type_ids' in input_dict:
        token_type_ids = input_dict['token_type_ids']
        token_type_ids = token_type_ids.to(device)
        model_kwargs['token_type_ids'] = token_type_ids

    return model_kwargs


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, device='cpu', gradual_unfreeze=False, 
                    save_likelihoods=None, hans=False):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        # self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradual_unfreeze = gradual_unfreeze
        self.device = device
        # self.batch_idx = 0
        self.epoch = 0
        self.num_layers = len(list(self.model.modules()))
        self.test_every = 1000
        self.index = 0
        # model.to(self.device)
        self.save_likelihoods = save_likelihoods 
        if self.save_likelihoods is not None:
            self.likelihoods = []
        self.hans = hans

    @staticmethod
    def _loss_mean(loss, attention):
        assert loss.shape == attention.shape, f"loss and attention must have the same\
            shape, but loss shape is {loss.shape} and attention shape is {attention.shape}"

        mean = (loss * attention).sum(1) / attention.sum(1)

        return mean

    @staticmethod
    def _reduce(loss, attention=None, reduction='mean', dim=-1):
        if reduction == 'mean':
            if attention is not None:
                loss = Trainer._loss_mean(loss, attention)
            else:
                loss = loss.mean(dim)
        elif reduction == 'sum':
            loss = loss.sum(dim)

        return loss

    def _get_ids_for_mnli(self,path):
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            lines = f.readlines()
            res = [int(l.strip()) for l in lines]
        self.mnli_ids = res
        return res

    def freeze_remaining_layers(self):
        for idx, layer in enumerate(self.model.modules()):
            if idx > self.num_layers * self.freeze_ratio:
                return
            layer.requires_grad = False

    def unfreeze_all(self):
        for layer in self.model.modules():
            layer.requires_grad = True

    def freeze_all(self):
        for layer in self.model.modules():
            layer.requires_grad = False

    def unfreeze_one_layer(self):
        last_layer = None
        for layer in self.model.modules():
            if layer.requires_grad == True and last_layer is not None:
                last_layer.requires_grad = True
                return
            last_layer = layer
        last_layer.requires_grad = True

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, writer=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        model_filename = None
        if checkpoints is not None:
            drive = kw.pop('drive', False)
            if drive:
                checkpoint_filename = f'/content/drive/My Drive/PremiseGeneratorBert/{checkpoints}.pt'
                model_filename = f'/content/drive/My Drive/PremiseGeneratorBert/{checkpoints}_model'
            else:
                checkpoint_filename = f'{checkpoints}.pt'
                model_filename = f'{checkpoints}_model'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement = \
                    saved_state.get('ewi', epochs_without_improvement)
                train_loss = saved_state.get('train_loss', train_loss)
                train_acc = saved_state.get('train_acc', train_acc)
                test_loss = saved_state.get('test_loss', test_loss)
                test_acc = saved_state.get('test_acc', test_acc)
                # writer = saved_state.get('writer', writer)
                actual_num_epochs = saved_state.get('ane', actual_num_epochs)
                # print(f"Loading model from {model_filename}")
                # from transformers import AutoModel, EncoderDecoderModel
                # # self.model = AutoModel.from_pretrained(model_filename)
                # self.model = EncoderDecoderModel.from_pretrained(model_filename)
                # self.model.to(self.device)
        kw.pop('drive', None)

        if self.gradual_unfreeze:
            self.freeze_all()

        while actual_num_epochs < num_epochs:

            if self.gradual_unfreeze:
                self.unfreeze_one_layer()
            epoch = actual_num_epochs
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            if not best_acc:
                best_acc = -10e8

            def freeze_remaining_layers(params, unfreezing_ratio):
                params = list(params)
                num_params = len(params)
                for idx, param in enumerate(params):
                    if idx > num_params * unfreezing_ratio:
                        return
                    param.requires_grad = False

            # unfreezing_one_layer(self.model.encoder.parameters())
            # unfreezing_one_layer(self.model.decoder.parameters())
            # unfreezing_one_layer(self.model.parameters())
            ######################################################
            train_result = self.train_epoch(dl_train, **kw)
            train_loss.append(torch.tensor(train_result.losses).mean().item())
            train_acc.append(train_result.accuracy)

            if writer is not None:
                writer.add_scalar('Loss/train', train_loss[-1], epoch)
                writer.add_scalar('Accuracy/train', train_acc[-1], epoch)

            test_result = self.test_epoch(dl_test, **kw)
            test_loss.append(torch.tensor(test_result.losses).mean().item())
            test_acc.append(test_result.accuracy)

            if writer is not None:
                writer.add_scalar('Loss/test', test_loss[-1], epoch)
                writer.add_scalar('Accuracy/test', test_acc[-1], epoch)

            if early_stopping and test_acc[-1] <= best_acc:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping:
                    break
            else:
                epochs_without_improvement = 0

            if checkpoints is not None and test_acc[-1] > best_acc:
                save_checkpoint = True
                best_acc = test_acc[-1]

            if test_acc[-1] > best_acc:
                best_acc = test_acc[-1]

            actual_num_epochs += 1
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   #    model_state=self.model.state_dict(),
                                   train_loss=train_loss,
                                   train_acc=train_acc,
                                   test_loss=test_loss,
                                   test_acc=test_acc,
                                   #    writer=writer,
                                   ane=actual_num_epochs
                                   )
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch + 1}')
                # self.model.save_pretrained(model_filename)
                if not os.path.isdir(model_filename):
                    os.makedirs(model_filename)

                self.model.save_pretrained(model_filename)
                print(f'*** Saved model {model_filename} ')

                if hasattr(self,'hyp_prior_model') and self.hyp_prior_model is not None and self.hyp_prior_model.requires_grad_:
                    self.hyp_prior_model.save_pretrained(f'{model_filename}_disc_prior')
                    print(f'*** Saved model {model_filename}_disc_prior ')
                if writer is not None:
                    writer.add_scalar('Loss/train', train_loss[-1], epoch)
                    writer.add_scalar('Accuracy/train', train_acc[-1], epoch)
            
            if checkpoint_filename is not None and test_loss[-1] <= min(test_loss) and hasattr(self,'gamma') and self.gamma>0.0:
                self.model.save_pretrained(f'{model_filename}_min_loss')
                print(f'*** Saved model {model_filename}_min_loss ')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def test(self, dl_test: DataLoader, checkpoints=None, post_epoch_fn=None, writer=None, **kw) -> FitResult:

        test_loss, test_acc = [], []

        verbose = False  # pass this to train/test_epoch.

        if checkpoints is not None:
            drive = kw.pop('drive', False)
            if drive:
                checkpoint_filename = f'/content/drive/My Drive/PremiseGeneratorBert/{checkpoints}.pt'
                model_filename = f'/content/drive/My Drive/PremiseGeneratorBert/{checkpoints}_model'
            else:
                checkpoint_filename = f'{checkpoints}.pt'
                model_filename = f'{checkpoints}_model'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                # best_acc = saved_state.get('best_acc', best_acc)
                # epochs_without_improvement = \
                #     saved_state.get('ewi', epochs_without_improvement)
                # train_loss = saved_state.get('train_loss', train_loss)
                # train_acc = saved_state.get('train_acc', train_acc)
                test_loss = saved_state.get('test_loss', test_loss)
                test_acc = saved_state.get('test_acc', test_acc)
                writer = saved_state.get('writer', writer)
                # self.model.load_state_dict(saved_state['model_state'])

        test_result = self.test_epoch(dl_test, **kw)
        test_loss.append(torch.tensor(test_result.losses).mean().item())
        test_acc.append(test_result.accuracy)

        if writer is not None:
            writer.add_scalar('Loss/test', test_loss[-1], 0)
            writer.add_scalar('Accuracy/test', test_acc[-1], 0)
        # ========================

        if post_epoch_fn:
            post_epoch_fn(0, EpochResult([0], 0), test_result, verbose)

        if self.save_likelihoods is not None:
            torch.save(torch.cat(self.likelihoods), f'{self.save_likelihoods}.torch')
            self.likelihoods = []

        return FitResult(0,
                         0, 0, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()  # set train mode
        if hasattr(self, 'eval_every'):
            kw['eval_every'] = self.eval_every
        return self._foreach_batch(dl_train, self.train_batch, mode='train', **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, mode='test', **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None, mode='train', eval_every=1) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        global return_acc
        losses = []
        num_correct = 0
        num_eval_batches = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        skip=0
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                # if len(data[0][0])>6000:
                #     skip+=1
                #     continue
                if (batch_idx + 1) % eval_every == 0:
                    return_acc = True
                    num_eval_batches += 1

                try:
                    batch_res = forward_fn(data)
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    batch_res = forward_fn(data)
                # import pdb;pdb.set_trace()
                num_correct += batch_res.num_correct
                running_acc = num_correct / ((num_eval_batches if num_eval_batches > 0 else 1) * dl.batch_size) * 100.0
                pbar.set_description(
                    f'{pbar_name} (loss:{batch_res.loss:.3f}, accuracy:{running_acc:.2f})'
                )
                pbar.update()
                if type(batch_res.loss) == float:
                    losses.append(batch_res.loss)
                else:
                    loss_item = batch_res.loss.item()
                    losses.append(loss_item)

                return_acc = False
            
            # import pdb; pdb.set_trace()

            avg_loss = sum(losses) / num_batches
            num = num_samples - skip if eval_every == 1 or mode == 'test' else num_eval_batches * dl.batch_size
            # num = num_samples if mode == 'test' else (num_samples / num_batches)
            accuracy = 100. * num_correct / num
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.2f})')

        return EpochResult(losses=losses, accuracy=accuracy)


class GenerativeTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, max_len=128, possible_labels_ids=None,
                 epsilon=0.0, tokenizer_encoder=None, tokenizer_decoder=None,
                 create_premises=False, gradual_unfreeze=False, clip=False, gamma=0.0,
                 rev=0.0, save_results=None, reduction='mean', hyp_prior_model=None, 
                 mnli_ids_path=None, decoder_only=False, hyp_weight=None, test_with_prior=False, device=None, ratios=None, 
                 save_likelihoods=None, generate_all_labels=False, hans=False, **kwargs):
        super().__init__(model, None, optimizer, scheduler, device=device, gradual_unfreeze=gradual_unfreeze, save_likelihoods=save_likelihoods,hans=hans)
        # self.evaluator = evaluator
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder if tokenizer_decoder is not None else tokenizer_encoder
        self.max_len = max_len
        self.last_freeze_loss = None
        self.freeze_ratio = 0.0
        self.num_layers = len(list(self.model.modules()))
        self.labels = possible_labels_ids
        self.epsilon = epsilon
        self.create_premises = create_premises
        self.gamma = gamma
        self.clip = clip
        self.save_results = save_results
        self.rev = rev
        self.num_labels = len(self.labels)
        self.reduction = reduction
        self.hyp_prior_model = hyp_prior_model
        self.eval_every = 50
        if self.save_results is not None and mnli_ids_path is not None:
            self._get_ids_for_mnli(mnli_ids_path)
        self.decoder_only = decoder_only
        self.test_with_prior = test_with_prior
        self.ratios = ratios
        self.generate_all_labels = generate_all_labels

    def _prepare_batch(self, batch):
        if len(batch)==2:
            PH, labels = batch
            PH, labels = list(PH), list(labels)
            labels = torch.tensor([self.labels[l] for l in labels]).unsqueeze(-1)
            dec_out = self.tokenizer_decoder.batch_encode_plus(PH, padding='longest', return_tensors='pt', truncation=True)
            batch = labels, torch.ones_like(labels), \
                dec_out['input_ids'].to(self.device), dec_out['attention_mask'].to(self.device)
            return batch
        P, H, labels = batch[0:3]
        P, H, labels = list(P), list(H), list(labels)
        if labels is None:
            labels = [0 for _ in range(len(H))]
        input_dict_encoder = self.tokenizer_encoder.batch_encode_plus(H, padding='longest', return_tensors='pt', truncation=True)
        x = input_dict_encoder['input_ids']
        bos = x[:, 0]
        rest = x[:, 1:]
        # import pdb; pdb.set_trace()
        labels_tokens = torch.tensor([self.labels[l] for l in labels])
        input_dict_encoder['input_ids'] = \
            torch.cat([bos.unsqueeze(0), labels_tokens.unsqueeze(0), rest.T]).T

        input_dict_encoder['attention_mask'] = \
            torch.cat([torch.ones(size=(1, x.shape[0]), dtype=torch.int), input_dict_encoder['attention_mask'].T]).T

        input_dict_decoder = self.tokenizer_decoder.batch_encode_plus(P, padding='longest', return_tensors='pt', truncation=True)

        batch = input_dict_encoder['input_ids'].to(self.device), input_dict_encoder['attention_mask'].to(self.device), \
                input_dict_decoder['input_ids'].to(self.device), input_dict_decoder['attention_mask'].to(self.device)

        return batch

    def _prepare_batch_disc(self, batch):
        input_dict = {}
        labels = None
        if len(batch) == 3:  # P, H, y
            P, H, labels = batch[0:3]
            P, H = list(P), list(H)
            input_dict = self.tokenizer_decoder.batch_encode_plus([[P[i], H[i]] for i in range(len(P))], padding='longest',
                                                          return_tensors='pt', truncation=True)
        elif len(batch) == 2:  # Hypotesis only
            H, labels = batch
            H = list(H)
            input_dict = self.tokenizer_decoder.batch_encode_plus(H, padding='longest', return_tensors='pt', truncation=True)

        batch_encoded = [input_dict[item].to(self.hyp_prior_model.device) for item in ['input_ids', 'attention_mask']]
        batch_encoded += [labels.to(self.hyp_prior_model.device)]

        if 'token_type_ids' in input_dict:
            token_type_ids = input_dict['token_type_ids']
            batch_encoded += [token_type_ids.to(self.hyp_prior_model.device)]
        else:
            batch_encoded += [None]

        return batch_encoded

    def _prepare_batch_decoder_only(self, batch):
        input_dict = {}
        if len(batch) == 3:
            P, H, labels = batch[0:3]
            P, H = list(P), list(H)
            y = [self.tokenizer_decoder.decode(self.labels[l]) for l in labels]
            input_dict = self.tokenizer_decoder.batch_encode_plus([[f"{y[i]} {H[i]}",P[i]] for i in range(len(P))], padding='longest',
                                                            return_tensors='pt', truncation=True)
        else: 
            H, labels = batch
            H = list(H)
            y = [self.tokenizer_decoder.decode(self.labels[l]) for l in labels]
            input_dict = self.tokenizer_decoder.batch_encode_plus([f"{y[i]} {H[i]}" for i in range(len(H))], padding='longest',
                                                            return_tensors='pt', truncation=True)

        batch_encoded = [input_dict[item].to(self.device) for item in ['input_ids', 'attention_mask']]

        if 'token_type_ids' in input_dict:
            token_type_ids = input_dict['token_type_ids']
            batch_encoded += [token_type_ids.to(self.device)]
        else:
            batch_encoded += [None]

        return batch_encoded

    def _next_label(self,l,delta=1):
        idx = (self.labels.index(l) + delta) % len(self.labels)
        return self.labels[idx]

    def train_epoch(self, dl_train: DataLoader, **kw):
        if self.hyp_prior_model is not None:
            self.hyp_prior_model.train()
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        if self.hyp_prior_model is not None:
            self.hyp_prior_model.eval()
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        if self.epsilon > 0.0:
            self.train_batch_label_smoothing(batch)
        elif self.decoder_only:
            return self.train_batch_decoder(batch)
        # elif self.gamma > 0.0:
        #     return self.train_batch_joint(batch)
        else:
            return self.train_batch_normal(batch)

    def train_batch_normal(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = self._prepare_batch(batch)
        x = x.to(self.device)
        y = y.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        decoder_attention_mask = decoder_attention_mask.to(self.device)
        num_correct = 0
        batch_size = x.shape[0]

        if x.size(1)>1 and x[0, 1] in self.labels:
            label_loc = 1
        else:
            label_loc = 0

        ## Needed because that gpt2 handles paddings differently
        mask = (decoder_attention_mask == 1)
        labels = y.clone() * mask + -100 * ~mask
        decoder_input_ids = y
        if self.model.config.architectures is not None and (any('Bart' in elem for elem in self.model.config.architectures) or any('T5' in elem for elem in self.model.config.architectures)):
                decoder_input_ids = None
                # labels = y

        model_kwargs = {
            "attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels
        }
        if decoder_input_ids is not None:
            model_kwargs['decoder_input_ids'] = decoder_input_ids
        else:
            model_kwargs.pop('decoder_attention_mask', None)
        self.optimizer.zero_grad()

        # prior = -torch.log(torch.tensor(1/self.num_labels))  
        outputs = self.model(input_ids=x, **model_kwargs)
        loss = outputs[0]
        loss = loss.view(batch_size, -1)
        # import pdb; pdb.set_trace()
        attention = decoder_attention_mask[:,
                    1:] if loss.shape != decoder_attention_mask.shape else decoder_attention_mask
        loss = self._reduce(loss, attention, reduction=self.reduction)
        if self.ratios is not None:
            # import pdb; pdb.set_trace()
            rates = torch.tensor([self.ratios[i] for i in batch[2]],device=self.device)
            rates = -(rates).log()
            loss = loss + rates
        # if self.hyp_prior_model is not None:
        #     # outputs_hyp = self.model(input_ids=x, **model_kwargs)
        #     prior = self.calc_disc_loss(batch)
        #     loss += prior
        if self.hyp_prior_model is not None:
            prior = self.calc_disc_loss(batch)
            loss += prior

        if self.gamma > 0.0:  ## Joint
            losses = [loss.clone()]
            for delta in range(1, len(self.labels)):
                bad_x = x.clone().to(self.device)
                # labels_tokens = ((bad_x[:, 1] - min_label + delta) % (len(self.labels))) + min_label # a very complicated way to change all the labels
                # import pdb; pdb.set_trace()
                labels_tokens = torch.tensor([self._next_label(l,delta) for l in bad_x[:, label_loc]])
                bad_x[:, label_loc] = labels_tokens
                bad_out = self.model(input_ids=bad_x, **model_kwargs)
                bad_loss = bad_out[0]
                bad_loss = bad_loss.view(batch_size, -1)
                bad_loss = self._reduce(bad_loss, attention, reduction=self.reduction)
                # import pdb; pdb.set_trace()
                if self.hyp_prior_model is not None:
                    bad_labels = (batch[2] + delta) % self.num_labels
                    batch_batch = (None, batch[1], bad_labels)
                    bad_prior = self.calc_disc_loss(batch_batch)
                    bad_loss += bad_prior

                losses.append(bad_loss)

            neg_log_prob = torch.stack(losses, 1)
            log_prob = -neg_log_prob
            log_sum_exp_losses = torch.logsumexp(log_prob, 1)
            loss = loss + self.gamma * log_sum_exp_losses
            # loss = self._reduce(loss, attention=None, reduction=self.reduction)

        
        
        loss_obj = self._reduce(loss, attention=None, reduction='mean')

        # torch.cuda.empty_cache()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        loss_obj.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        del x, y, encoder_attention_mask, decoder_attention_mask, labels
        # if self.gamma > 0.0 and self.rev:
        #     del rev_x, rev_y, rev_encoder_attention_mask, rev_decoder_attention_mask

        # validate on train set
        # num_correct = sum(loss < min_loss) if min_loss is not None else 0
        # if self.gamma == 0.0:
        num_correct = 0
        global mode, return_acc
        if return_acc:
            mode = 'train'
            with torch.no_grad():
                self.model.eval()  # small hack but it's working
                acc = self.test_batch(batch)
                num_correct = acc.num_correct
                self.model.train()
            mode = 'test'

        # num_correct=0

        loss_item = loss_obj.item()

        if self.freeze_ratio > 0.0:
            if self.last_freeze_loss is None:
                self.last_freeze_loss = loss_item
            elif loss <= 0.5 * self.last_freeze_loss:
                self.last_freeze_loss = loss_item
                print("Freezing half of the unfrozzen layers")
                self.freeze_remaining_layers()
                self.freeze_ratio = 0.5 * self.freeze_ratio + 0.5

        return BatchResult(loss_item, num_correct)

    def train_batch_decoder(self, batch) -> BatchResult:  
        # import pdb; pdb.set_trace()
        x, attention_mask, type_ids = self._prepare_batch_decoder_only(batch)
        x = x.to(self.device)   # y hypothesis premise
        attention_mask = attention_mask.to(self.device)
        type_ids = type_ids.to(self.device) if type_ids is not None else type_ids

        _, y_hyp_mask, _ = self._prepare_batch_decoder_only(batch[1:])
        y_hyp_mask = y_hyp_mask.to(self.device)
        
        premise_mask = torch.zeros_like(attention_mask)
        premise_mask[:,:y_hyp_mask.size(1)]=y_hyp_mask
        premise_mask = attention_mask - premise_mask

        num_correct = 0
        batch_size = x.shape[0]

        if x[0, 1] in self.labels:
            label_loc = 1
        else:
            label_loc = 0

        ## Needed because that gpt2 handles paddings differently
        mask = (attention_mask == 1)
        labels = x.clone() * mask + -100 * ~mask

        model_kwargs = {
            "input_ids": x,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
            "labels": labels,
        }
        
        self.optimizer.zero_grad()

        # prior = -torch.log(torch.tensor(1/self.num_labels))
        outputs = self.model(**model_kwargs)

        loss = outputs[0]
        loss = loss.view(batch_size, -1)
        attention = premise_mask
        attention = attention[:, 1:] if loss.shape != attention.shape else attention
        # import pdb; pdb.set_trace()
        loss = self._reduce(loss, attention, reduction=self.reduction)

        if self.hyp_prior_model is not None:
            prior = self.calc_disc_loss(batch)
            loss += prior

        if self.gamma > 0.0:  ## Joint
            model_kwargs.pop('input_ids',None)
            losses = [loss.clone()]
            for delta in range(1, len(self.labels)):
                bad_x = x.clone().to(self.device)
                labels_tokens = torch.tensor([self._next_label(l,delta) for l in bad_x[:, label_loc]])
                bad_x[:, label_loc] = labels_tokens
                bad_out = self.model(input_ids=bad_x, **model_kwargs)
                bad_loss = bad_out[0]
                bad_loss = bad_loss.view(batch_size, -1)
                bad_loss = self._reduce(bad_loss, attention, reduction=self.reduction)
                if self.hyp_prior_model is not None:
                    bad_labels = (batch[2] + delta) % self.num_labels
                    batch_batch = (None, batch[1], bad_labels)
                    bad_prior = self.calc_disc_loss(batch_batch)
                    bad_loss += bad_prior

                losses.append(bad_loss)

            neg_log_prob = torch.stack(losses, 1)
            log_prob = -neg_log_prob
            log_sum_exp_losses = torch.logsumexp(log_prob, 1)
            loss = loss + self.gamma * log_sum_exp_losses
        
        
        loss_obj = self._reduce(loss, attention=None, reduction='mean')

        # torch.cuda.empty_cache()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        loss_obj.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        del x, attention_mask, labels
        num_correct = 0
        global mode, return_acc
        if return_acc:
            mode = 'train'
            with torch.no_grad():
                self.model.eval()  # small hack but it's working
                acc = self.test_batch(batch)
                num_correct = acc.num_correct
                self.model.train()
            mode = 'test'

        # num_correct=0

        loss_item = loss_obj.item()

        if self.freeze_ratio > 0.0:
            if self.last_freeze_loss is None:
                self.last_freeze_loss = loss_item
            elif loss <= 0.5 * self.last_freeze_loss:
                self.last_freeze_loss = loss_item
                print("Freezing half of the unfrozzen layers")
                self.freeze_remaining_layers()
                self.freeze_ratio = 0.5 * self.freeze_ratio + 0.5

        return BatchResult(loss_item, num_correct)

    def train_batch_label_smoothing(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = self._prepare_batch(batch)

        n_labels = len(self.labels)

        correct_labels = x[:, 1].clone()
        total_loss = torch.zeros(1, dtype=float)
        pred = []

        batch_size = x.size(0)

        inp_x = []
        inp_y = []
        inp_e_a_m = []
        inp_d_a_m = []
        # inp_d_e_a_m = []

        for label_id in self.labels:
            curr_x = x.clone()
            curr_x[:, 1] = label_id
            inp_x.append(curr_x)
            inp_y.append(y)
            inp_e_a_m.append(encoder_attention_mask)
            inp_d_a_m.append(decoder_attention_mask)
            # inp_d_e_a_m.append(decoder_encoder_attention_mask)

        inp_x = torch.cat(inp_x)
        inp_y = torch.cat(inp_y)
        inp_e_a_m = torch.cat(inp_e_a_m)
        inp_d_a_m = torch.cat(inp_d_a_m)
        # inp_d_e_a_m = torch.cat(inp_d_e_a_m)

        inp_x = inp_x.to(self.device)
        inp_y = inp_y.to(self.device)
        inp_e_a_m = inp_e_a_m.to(self.device)
        inp_d_a_m = inp_d_a_m.to(self.device)
        # inp_d_e_a_m = inp_d_e_a_m.to(self.device)

        model_kwargs = {
            "attention_mask": inp_e_a_m,
            "decoder_attention_mask": inp_d_a_m,
            "labels": inp_y
        }

        self.optimizer.zero_grad()

        outputs = self.model(input_ids=inp_x, decoder_input_ids=inp_y, **model_kwargs)
        loss = outputs[0]
        loss = loss.view(inp_x.size(0), -1)
        loss = loss.mean(dim=1)

        ret = torch.min(loss.view(n_labels, -1), dim=0)
        pred = ret.indices
        losses = ret.values

        # loss_obj = LabelSmoothingCrossEntropy(epsilon=self.epsilon)(-loss.view(-1,len(self.labels)),(correct_labels-min(self.labels)).to(self.device))

        l_idx = correct_labels - min(self.labels)  # get locations instead of tokens
        l_idx = l_idx.to(self.device)
        loss_obj = (1.0 - self.epsilon) * loss.view(n_labels, -1).T.gather(1, l_idx.view(-1, 1))  # smoothing

        for i in range(1, n_labels):
            indices = (l_idx + i) % n_labels  # next location
            loss_obj += (self.epsilon / n_labels) * loss.view(n_labels, -1).T.gather(1, indices.view(-1, 1))
        loss_obj[loss_obj < 0.0] = 0.0  # no negative
        loss_obj = loss_obj.mean()

        loss_obj.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # del x, y, encoder_attention_mask, decoder_attention_mask
        del inp_x, inp_y, inp_d_a_m, inp_e_a_m
        # torch.cuda.empty_cache()

        # total_loss = torch.sum(torch.tensor([l[0] for l in best_res]))
        total_loss = losses.sum()

        # pred = torch.tensor([label[1] for label in best_res])
        pred = torch.tensor([self.labels[i] for i in pred])
        pred.to('cpu')
        correct_labels = correct_labels.to('cpu')
        num_correct = torch.sum(pred == correct_labels).type(torch.FloatTensor)

        tot = loss_obj.item()

        return BatchResult(tot, num_correct.item())

    def train_batch_joint(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = self._prepare_batch(batch)
        if x[0, 1] in self.labels:
            label_loc = 1
        else:
            label_loc = 0
        correct_labels = x[:, label_loc].clone()
        pred = []

        inp_x = []
        inp_y = []
        inp_e_a_m = []
        inp_d_a_m = []
        # inp_d_e_a_m = []

        for label_id in self.labels:
            curr_x = x.clone()
            curr_x[:, label_loc] = label_id
            inp_x.append(curr_x)
            inp_y.append(y)
            inp_e_a_m.append(encoder_attention_mask)
            inp_d_a_m.append(decoder_attention_mask)

        inp_x = torch.cat(inp_x)
        inp_y = torch.cat(inp_y)
        inp_e_a_m = torch.cat(inp_e_a_m)
        inp_d_a_m = torch.cat(inp_d_a_m)

        inp_x = inp_x.to(self.device)
        inp_y = inp_y.to(self.device)
        inp_e_a_m = inp_e_a_m.to(self.device)
        inp_d_a_m = inp_d_a_m.to(self.device)

        ## To avoid computing loss for padding
        mask = (inp_d_a_m == 1)
        labels = inp_y.clone() * mask + -100 * ~mask

        model_kwargs = {
            "attention_mask": inp_e_a_m,
            "decoder_attention_mask": inp_d_a_m,
            "labels": labels
        }
        self.optimizer.zero_grad()
        outputs = self.model(input_ids=inp_x, decoder_input_ids=inp_y, **model_kwargs)
        loss = outputs[0]
        # import pdb; pdb.set_trace()
        loss = loss.view(inp_x.size(0), -1)
        attention = inp_d_a_m[:, 1:] if loss.shape != inp_d_a_m.shape else inp_d_a_m
        loss = self._reduce(loss, attention=attention, reduction=self.reduction)

        num_labels = len(self.labels)
        batch_size = x.shape[0]

        mask = (torch.arange(num_labels).view(-1, 1).repeat(1, batch_size) == (
                correct_labels.view(1, -1).repeat(num_labels, 1) - min(self.labels)))
        loss = loss.view(num_labels, -1)
        all_losses = loss.clone()
        total_loss = (loss * mask.to(self.device)).sum(0)
        total_loss += self.gamma * torch.logsumexp(-loss, 0)

        total_loss = self._reduce(total_loss, reduction=self.reduction, dim=0)
        total_loss.backward()

        del inp_x, inp_y, inp_d_a_m, inp_e_a_m, labels

        ret = torch.min(all_losses, dim=0)
        pred = ret.indices
        pred = torch.tensor([self.labels[i] for i in pred])
        pred.to('cpu')
        correct_labels = correct_labels.to('cpu')
        num_correct = torch.sum(pred == correct_labels).type(torch.FloatTensor)

        tot = total_loss.item()

        return BatchResult(tot, num_correct.item())

    def calc_disc_loss(self, batch):
        batch_hyp_only = (batch[1], batch[2])  # H, y
        y_hyp, attention_mask_hyp, labels_hyp, token_type_ids_hyp = self._prepare_batch_disc(batch_hyp_only)
        y_hyp = y_hyp.to(self.hyp_prior_model.device)
        attention_mask_hyp = attention_mask_hyp.to(self.hyp_prior_model.device)
        # labels_hyp = labels.to(self.device)
        # labels_hyp.to(self.device)
        args = {
            'input_ids':y_hyp,
            'attention_mask':attention_mask_hyp,
            'labels':batch[2].to(self.hyp_prior_model.device)
        }
        if token_type_ids_hyp is not None:
            token_type_ids_hyp = token_type_ids_hyp.to(self.hyp_prior_model.device)
            args['token_type_ids']=token_type_ids_hyp
        # import pdb; pdb.set_trace()
        prior = self.hyp_prior_model(**args)
        prior = prior[0]
        prior = prior.to(self.device)
        return prior

    def test_batch(self, batch) -> BatchResult:
        # import pdb; pdb.set_trace()
        if not self.decoder_only:
            inp_x = []
            inp_y = []
            inp_e_a_m = []
            inp_d_a_m = []
            if type(batch[0][0])!=tuple:
                x, encoder_attention_mask, y, decoder_attention_mask = self._prepare_batch(batch)
                if x.size(1)>1 and x[0, 1] in self.labels:
                    label_loc = 1
                else:
                    label_loc = 0
                correct_labels = x[:, label_loc].clone()
                total_loss = torch.zeros(1, dtype=float)
                pred = []

                batch_size = x.size(0)
                # inp_d_e_a_m = []

                for label_id in self.labels:
                    curr_x = x.clone()
                    curr_x[:, label_loc] = label_id
                    inp_x.append(curr_x)
                    inp_y.append(y)
                    inp_e_a_m.append(encoder_attention_mask)
                    inp_d_a_m.append(decoder_attention_mask)
            
            else:
                batch_size = len(batch[0][0])
                # import pdb; pdb.set_trace()
                correct_labels = torch.tensor([self.labels[l] for l in batch[2]])
                for i in range(len(self.labels)):
                    temp_labels = torch.zeros(batch_size,dtype=int)+i
                    temp_batch = ((batch[0][i]), batch[1][i], temp_labels)
                    x, encoder_attention_mask, y, decoder_attention_mask = self._prepare_batch(temp_batch)
                    inp_x.append(x)
                    inp_y.append(y)
                    inp_e_a_m.append(encoder_attention_mask)
                    inp_d_a_m.append(decoder_attention_mask)

                for l, val in [(inp_x,self.tokenizer_encoder.pad_token_id),(inp_y,self.tokenizer_decoder.pad_token_id),(inp_e_a_m,0),(inp_d_a_m,0)]:
                    # import pdb; pdb.set_trace()
                    if len(set([i.size(1) for i in l]))==1:
                        continue
                    max_len = max([i.size(1) for i in l])
                    for i,inp in enumerate(l):
                        new_inp = torch.zeros((batch_size,max_len),dtype=int)
                        new_inp[:,:inp.size(1)] = inp
                        new_inp[:,inp.size(1):] = val
                        l[i] = new_inp

                    
            inp_x = torch.cat(inp_x)
            inp_y = torch.cat(inp_y)
            inp_e_a_m = torch.cat(inp_e_a_m)
            inp_d_a_m = torch.cat(inp_d_a_m)

            inp_x = inp_x.to(self.device)
            inp_y = inp_y.to(self.device)
            inp_e_a_m = inp_e_a_m.to(self.device)
            inp_d_a_m = inp_d_a_m.to(self.device)
            ## Needed because that gpt2 handles paddings differently
            mask = (inp_d_a_m == 1)
            labels = inp_y.clone() * mask + -100 * ~mask
            decoder_input_ids = inp_y
            if self.model.config.architectures is not None and (any('Bart' in elem for elem in self.model.config.architectures) or any('T5' in elem for elem in self.model.config.architectures)):
                decoder_input_ids = None
                # labels = inp_y

            model_kwargs = {
                "input_ids":inp_x,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": inp_e_a_m,
                "decoder_attention_mask": inp_d_a_m,
                "labels": labels
            }

            attention = inp_d_a_m

        else: ##  if self.decoder_only:
            x, attention_mask, type_ids = self._prepare_batch_decoder_only(batch)

            _, y_hyp_mask, _ = self._prepare_batch_decoder_only(batch[1:])
            y_hyp_mask = y_hyp_mask.to(self.device)
            
            premise_mask = torch.zeros_like(attention_mask)
            premise_mask[:,:y_hyp_mask.size(1)]=y_hyp_mask
            premise_mask = attention_mask - premise_mask

            batch_size = x.shape[0]
            if x[0, 1] in self.labels:
                label_loc = 1
            else:
                label_loc = 0
            correct_labels = x[:, label_loc].clone()
            total_loss = torch.zeros(1, dtype=float)
            pred = []

            batch_size = x.size(0)

            inp_x = []
            inp_a_m = []
            inp_t_t = []
            premise_mask_extended = []
            # import pdb; pdb.set_trace()

            for label_id in self.labels:
                curr_x = x.clone()
                curr_x[:, label_loc] = label_id
                inp_x.append(curr_x)
                inp_a_m.append(attention_mask)
                if type_ids is not None:
                    inp_t_t.append(type_ids)
                premise_mask_extended.append(premise_mask)

            inp_x = torch.cat(inp_x)
            inp_a_m = torch.cat(inp_a_m)
            if type_ids is not None:
                inp_t_t = torch.cat(inp_t_t)
            premise_mask_extended = torch.cat(premise_mask_extended)

            inp_x = inp_x.to(self.device)
            inp_a_m = inp_a_m.to(self.device)
            if type_ids is not None:
                inp_t_t = inp_t_t.to(self.device)
            premise_mask_extended = premise_mask_extended.to(self.device)

            ## Needed because that gpt2 handles paddings differently
            mask = (inp_a_m == 1)
            labels = inp_x.clone() * mask + -100 * ~mask

            model_kwargs = {
                "input_ids":inp_x,
                "attention_mask": inp_a_m,
                "labels": labels,
            }
            if type_ids is not None:
                model_kwargs["token_type_ids"] = inp_t_t
            
            attention = premise_mask_extended
        # import pdb; pdb.set_trace()
        num_labels = len(self.labels)

        with torch.no_grad():
            outputs = self.model(**model_kwargs)
            loss = outputs[0]
            loss = loss.view(inp_x.size(0), -1)
            attention = attention[:, 1:] if loss.shape != attention.shape else attention
            loss = self._reduce(loss, attention=attention, reduction=self.reduction)
            if self.hyp_prior_model is not None and self.test_with_prior:
                # pdb.set_trace()
                test_labels = torch.arange(self.num_labels).repeat(batch_size,1).T.reshape(-1)  # (0,...,0,1,...,1,2,...,2)
                hyp_batch_test = (None, batch[1]*3, test_labels)
                prior = self.calc_disc_loss(hyp_batch_test)
                loss += prior

            ret = torch.min(loss.view(len(self.labels), -1), dim=0)
            pred = ret.indices
            if mode == 'test':
                mask = (torch.arange(num_labels).view(-1, 1).repeat(1, batch_size).to(self.device) == (
                        correct_labels.view(1, -1).repeat(num_labels, 1) - min(self.labels)).to(self.device))
                loss = loss.view(num_labels, -1)
                total_loss = (loss * mask.to(self.device)).sum(0)
                # total_loss = ret.values
                if self.gamma > 0.0:
                    total_loss += self.gamma * torch.logsumexp(-loss, 0)
            else:
                total_loss = torch.tensor(0.0,dtype=torch.float)

            total_loss = total_loss.mean()
            pass
        if self.decoder_only:
            del inp_a_m, inp_t_t
        else:
            del inp_y, inp_d_a_m, inp_e_a_m
        del inp_x

        # self.hans = True
        if self.hans:
            # pdb.set_trace()
            pred[pred==0]=2
        pred = pred.to('cpu')
        # if batch[2] is None:
            # return pred
        correct_labels = batch[2].to('cpu')
        num_correct = torch.sum(pred == correct_labels).type(torch.FloatTensor)

        pred = torch.tensor([self.labels[i] for i in pred])

        # pdb.set_trace()
        if self.save_results is not None:
            if not os.path.isfile(self.save_results + '.csv'):
                with open(self.save_results + '.csv', 'w') as f:
                    f.write('pairID,gold_label\n')
            with open(self.save_results + '.csv', 'a') as f:
                for l in pred.tolist():
                    label = self.tokenizer_decoder.decode(self.labels[l])
                    label = label.replace('[', '').replace(']', '')
                    f.write(f'{self.mnli_ids[self.index]},{label}\n')
                    self.index += 1
        # pdb.set_trace()
        if self.save_likelihoods is not None:
            true_labels_loss = ((loss-(-(-loss).logsumexp(0)))* mask.to(self.device)).sum(0)
            # for l in true_labels_loss:
            self.likelihoods.append(pred == correct_labels)

        # for i in pred:
        #     temp[i-min(self.labels)]+=1
        # print(temp)

        tot = total_loss.item()

        return BatchResult(tot, num_correct.item())

    
    def generate_dataset(self,dl: DataLoader):
        self._foreach_batch(dl, self.generate_batch)

    
    def generate_batch(self, batch):
        inp_x = []
        inp_e_a_m = []
        
        x, encoder_attention_mask, _, _ = self._prepare_batch(batch)
        if x[0, 1] in self.labels:
            label_loc = 1
        else:
            label_loc = 0
        # pdb.set_trace()
        if self.generate_all_labels:
            for label_id in self.labels:
                curr_x = x.clone()
                curr_x[:, label_loc] = label_id
                inp_x.append(curr_x)
                inp_e_a_m.append(encoder_attention_mask)

            inp_x = torch.cat(inp_x)
            inp_e_a_m = torch.cat(inp_e_a_m)

            inp_x = inp_x.to(self.device)
            inp_e_a_m = inp_e_a_m.to(self.device)
        else:
            inp_x = x
            inp_e_a_m = encoder_attention_mask

        model_kwargs = {
            "input_ids":inp_x,
            "attention_mask": inp_e_a_m,
        }
        # import pdb; pdb.set_trace()
        summary_ids = self.model.generate(**model_kwargs)
        text = [self.tokenizer_encoder.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        with open(self.save_results + '_lbl_file', 'a') as f_lbl:
            with open(self.save_results + '_source_file', 'a') as f_src:
                for premise, hypothesis_tokens_ids, lbl in zip(text, inp_x[:,label_loc+1:], inp_x[:, label_loc]):
                    label = self.tokenizer_encoder.decode(lbl)[1:-1].lower()
                    # import pdb; pdb.set_trace()
                    hypothesis = self.tokenizer_encoder.decode(hypothesis_tokens_ids[hypothesis_tokens_ids>2], clean_up_tokenization_spaces=False)
                    f_src.write(f'{premise}|||{hypothesis}\n')
                    f_lbl.write(label+'\n')

        return BatchResult(0.0,0)

    # def perplexity(self, batch, stride=512):
    #     max_length = self.model.config.max_position_embeddings
    #     lls = []
    #     for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    #         begin_loc = max(i + stride - max_length, 0)
    #         end_loc = min(i + stride, encodings.input_ids.size(1))
    #         trg_len = end_loc - i    # may be different from stride on last loop
    #         input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    #         target_ids = input_ids.clone()
    #         target_ids[:,:-trg_len] = -100

    #         with torch.no_grad():
    #             outputs = model(input_ids, labels=target_ids)
    #             log_likelihood = outputs[0] * trg_len

    #         lls.append(log_likelihood)

    #     ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    #     return pll


class DiscriminativeTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, max_len=128, num_labels=3,
                 tokenizer=None, save_results=None, hyp_prior_model=None, mnli_ids_path=None, device=None, save_likelihoods=None, hans=False, **kwargs):
        super().__init__(model, None, optimizer=optimizer, scheduler=scheduler, device=device, save_likelihoods=save_likelihoods, hans=hans)
        # self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.last_freeze_loss = None
        self.freeze_ratio = 0.0
        self.labels = torch.tensor(range(num_labels))
        self.save_results = save_results
        self.hyp_prior_model = hyp_prior_model
        if self.hyp_prior_model is not None:
            self.hyp_prior_model.eval()
        if self.save_results is not None and mnli_ids_path is not None:
            self._get_ids_for_mnli(mnli_ids_path)

    def _prepare_batch(self, batch):
        input_dict = {}
        labels = None
        if len(batch) == 3:  # P, H, y
            P, H, labels = batch[0:3]
            input_dict = self.tokenizer.batch_encode_plus([[P[i], H[i]] for i in range(len(P))], padding='longest',
                                                          return_tensors='pt', truncation=True)
        elif len(batch) == 2:  # Hypotesis only
            H, labels = batch
            H = list(H)
            input_dict = self.tokenizer.batch_encode_plus(H, padding='longest', return_tensors='pt', truncation=True)

        batch_encoded = [input_dict[item].to(self.device) for item in ['input_ids', 'attention_mask']]
        batch_encoded += [labels.to(self.device)]

        if 'token_type_ids' in input_dict:
            token_type_ids = input_dict['token_type_ids']
            batch_encoded += [token_type_ids.to(self.device)]
        else:
            batch_encoded += [None]

        return batch_encoded

    def train_epoch(self, dl_train: DataLoader, **kw):
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, attention_mask, labels, token_type_ids = self._prepare_batch(batch)
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # token_type_ids = token_type_ids.to(self.device)
        labels = labels.to(self.device)

        model_kwargs = {
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "labels": labels
        }

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
            model_kwargs['token_type_ids'] = token_type_ids

        self.optimizer.zero_grad()
        # pdb.set_trace()
        outputs = self.model(input_ids=x, **model_kwargs)

        loss, logits = outputs[:2]

        if self.hyp_prior_model is not None:
            batch_hyp = (batch[1], batch[2])
            x_hyp, attention_mask_hyp, labels_hyp, token_type_ids_hyp = self._prepare_batch(batch_hyp)
            x_hyp, attention_mask_hyp, labels_hyp, token_type_ids_hyp = x_hyp.to(self.device), attention_mask_hyp.to(
                self.device), labels_hyp.to(self.device), token_type_ids_hyp.to(self.device)
            with torch.no_grad():
                prior = self.hyp_prior_model(input_ids=x_hyp, attention_mask=attention_mask_hyp, labels=labels_hyp,
                                             token_type_ids=token_type_ids_hyp)
            prior = prior[0]
            # import pdb; pdb.set_trace()
            loss *= prior

        loss = loss.mean()

        del x, attention_mask
        if token_type_ids is not None:
            del token_type_ids

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # validate
        labels = labels.to('cpu')
        pred = torch.argmax(logits, dim=1).to('cpu')
        # pdb.set_trace()

        num_correct = torch.sum(labels == pred)

        loss_item = loss.item()

        del loss

        return BatchResult(loss_item, num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        x, attention_mask, labels, token_type_ids = self._prepare_batch(batch)
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # token_type_ids = token_type_ids.to(self.device)
        labels = labels.to(self.device)

        model_kwargs = {
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "labels": labels
        }

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
            model_kwargs['token_type_ids'] = token_type_ids

        with torch.no_grad():
            outputs = self.model(input_ids=x, **model_kwargs)

            loss, logits = outputs[:2]
            loss = loss.mean()

        del x, attention_mask
        if token_type_ids is not None:
            del token_type_ids

        # check accuracy
        labels = labels.to('cpu')
        pred = torch.argmax(logits, dim=1).to('cpu')
        
        if self.hans:
            pred[pred==0] = 2
        # import pdb; pdb.set_trace()
        
        toks = self.tokenizer.batch_encode_plus(list(batch[1]), padding='longest', return_tensors='pt', truncation=True)
        first_tok = toks['input_ids'][:, 1]
        first_tok[first_tok == 7454] = 50001
        first_tok = first_tok-50000 # 0, 1 or 2
        first_tok = first_tok.cpu().numpy().tolist()

        if self.save_results is not None:
            if not os.path.isfile(self.save_results + '.csv'):
                with open(self.save_results + '.csv', 'w') as f:
                    f.write('pairID,gold_label\n')

            if not os.path.isfile(self.save_results + '_firsttoken.csv'):
                with open(self.save_results + '_firsttoken.csv', 'w') as f:
                    f.write('pairID,firsttoken\n')

            possible_labels = ['contradiction', 'entailment', 'neutral']
            with open(self.save_results + '_firsttoken.csv', 'a') as f_ft:
                with open(self.save_results + '.csv', 'a') as f:
                    for l, ft in zip(pred.tolist(), first_tok):
                        label = possible_labels[l]
                        ft = possible_labels[ft]
                        f.write(f'{self.mnli_ids[self.index]},{label}\n')
                        f_ft.write(f'{self.mnli_ids[self.index]},{ft}\n')
                        self.index += 1

        # pdb.set_trace()
        if self.save_likelihoods is not None:
            true_labels_loss = outputs[0]
            # true_labels_loss = torch.nn.LogSoftmax(1)(outputs[1])
            # for l in true_labels_loss:
            self.likelihoods.append(labels == pred)

        num_correct = torch.sum(labels == pred)

        loss_item = loss.item()

        # for i in pred:
        #     temp[i]+=1
        # print(temp)

        del loss
        return BatchResult(loss_item, num_correct.item())


class OnelabelTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, max_len=128, possible_labels_ids=None,
                 epsilon=0.0, tokenizer_encoder=None, tokenizer_decoder=None, create_premises=False,
                 label=0, save_results=None, device=None, **kwargs):
        super().__init__(model, None, optimizer, scheduler, device)
        # self.evaluator = evaluator
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder if tokenizer_decoder is not None else tokenizer_encoder
        self.max_len = max_len
        self.last_freeze_loss = None
        self.freeze_ratio = 0.0
        self.num_layers = len(list(self.model.modules()))
        self.labels = possible_labels_ids
        self.epsilon = epsilon
        self.create_premises = create_premises
        # self.rouge = nlp.load_metric("rouge", experiment_id=label)
        self.save_results = save_results
        self.label = label

    def _prepare_batch(self, batch):
        P, H, labels = batch[0:3]
        P, H, labels = list(P), list(H), list(labels)
        input_dict_encoder = self.tokenizer_encoder.batch_encode_plus(H, padding='longest', return_tensors='pt')
        input_dict_decoder = self.tokenizer_decoder.batch_encode_plus(P, padding='longest', return_tensors='pt')

        batch = (input_dict_encoder['input_ids'].to(self.device), input_dict_encoder['attention_mask'].to(self.device),
                 input_dict_decoder['input_ids'].to(self.device), input_dict_decoder['attention_mask'].to(self.device), labels.to(self.device))

        return batch

    def train_epoch(self, dl_train: DataLoader, **kw):
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask, labels = self._prepare_batch(batch)
        x = x.to(self.device)
        y = y.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        decoder_attention_mask = decoder_attention_mask.to(self.device)

        batch_size = x.shape[0]

        model_kwargs = {
            "attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": y
        }
        self.optimizer.zero_grad()

        outputs = self.model(input_ids=x, decoder_input_ids=y, **model_kwargs)
        loss = outputs[0]
        loss = loss.view(batch_size, -1)
        loss = self._loss_mean(loss, decoder_attention_mask[:, 1:])
        # loss[labels!=self.label] *= -0.01
        loss[labels != self.label] = 10 / loss[labels != self.label]
        loss_obj = loss.mean()

        del y, encoder_attention_mask, decoder_attention_mask

        loss_obj.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        del x

        loss_item = loss_obj.item()

        acc = 1.0 / loss_item

        return BatchResult(loss_item, acc)

    def test_batch(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask, labels = self._prepare_batch(batch)
        x = x.to(self.device)
        y = y.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        decoder_attention_mask = decoder_attention_mask.to(self.device)

        model_kwargs = {
            "attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": y.clone()
        }

        batch_size = x.shape[0]

        with torch.no_grad():
            outputs = self.model(input_ids=x, decoder_input_ids=y, **model_kwargs)
            loss = outputs[0]
            loss = loss.view(batch_size, -1)
            loss = self._loss_mean(loss, decoder_attention_mask[:, 1:])
            # loss[labels!=self.label] *= -0.01
            loss[labels!=self.label] = 10 / loss[labels!=self.label]
            if self.save_results is not None:
                with open(self.save_results, 'a') as f:
                    for l in loss.tolist():
                        f.write(f'{l}\n')

            loss = loss.mean()

        del x, y, encoder_attention_mask, decoder_attention_mask

        tot = loss.item()

        acc = 1.0 / tot  # minimize the loss on the validation set

        return BatchResult(tot, acc)
