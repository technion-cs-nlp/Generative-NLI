import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Any
from pathlib import Path
from src.utils import BatchResult, EpochResult, FitResult

from src.loss import LabelSmoothingCrossEntropy

# return_acc = False


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, device='cpu'):
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
        self.device = device
        # self.batch_idx = 0
        self.epoch = 0
        self.num_layers = len(list(self.model.modules()))
        # model.to(self.device)

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
                # self.model = AutoModel.from_pretrained(model_filename)
                # self.model.to(self.device)
        kw.pop('drive', None)

        while actual_num_epochs < num_epochs:
            epoch = actual_num_epochs
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            if not best_acc:
                best_acc = -1
            # Unfreezing one layer
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
                if writer is not None:
                    writer.add_scalar('Loss/train', train_loss[-1], epoch)
                    writer.add_scalar('Accuracy/train', train_acc[-1], epoch)

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def test(self, dl_test: DataLoader,checkpoints=None, post_epoch_fn=None, writer=None, **kw) -> FitResult:

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
                       verbose=True, max_batches=None, mode='train') -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        # global return_acc
        losses = []
        num_correct = 0
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
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                # if batch_idx == num_batches-1:         # calculate acuuracy for train each 1000 epochs
                #     return_acc = True

                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            # num = num_samples if mode == 'test' else (num_samples / num_batches)
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')
        # return_acc = False

        return EpochResult(losses=losses, accuracy=accuracy)


class PremiseGeneratorTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, max_len=128, possible_labels_ids=None,
                epsilon=0.0, device=None):
        super().__init__(model, None, optimizer, scheduler, device)
        # self.evaluator = evaluator
        # self.tokenizer = tokenizer
        self.max_len = max_len
        self.last_freeze_loss = None
        self.freeze_ratio = 0.0
        self.num_layers = len(list(self.model.modules()))
        self.labels = possible_labels_ids
        self.epsilon = epsilon

    def train_epoch(self, dl_train: DataLoader, **kw):
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        if self.epsilon > 0.0:
            return self.train_batch_label_smoothing(batch)
        else:
            return self.train_batch_normal(batch)

    def train_batch_normal(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = batch
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
        # import pdb; pdb.set_trace()
        loss = outputs[0]
        logits = outputs[1]
        loss = loss.view(-1,batch_size).mean(0)
        x.detach()
        min_loss = None
        if False:       ## Infrastracture
            max_label = max(self.labels)
            # max_loss = torch.zeros(batch_size, device=self.device,requires_grad=True)
            max_log = torch.zeros_like(logits, device=self.device,requires_grad=True)
            min_loss = torch.zeros(batch_size, device=self.device,requires_grad=True) + 1000.0
            for delta in range(1,len(self.labels)):
                bad_x = x.clone().to(self.device)
                bad_x[:, 1] = max_label - (max_label - bad_x[:, 1] + delta) % (len(self.labels))        # a very complicated way to change all the labels
                bad_out = self.model(input_ids=bad_x, decoder_input_ids=y, **model_kwargs)
                bad_loss = bad_out[0]
                bad_log = bad_out[1]
                bad_loss = bad_loss.view(-1,batch_size).mean(0)
                # max_loss = torch.max(max_loss, bad_loss)
                max_log = torch.max(max_log, bad_log)
                min_loss = torch.min(min_loss, bad_loss)
                del bad_x
            # import pdb; pdb.set_trace()
            # diff = (loss - min_loss)
            diff = (max_log - logits)
            zero = torch.zeros_like(diff, device=self.device,requires_grad=True)
            loss_obj = torch.max(zero, 1.0 + diff)
            loss_obj = loss_obj.mean()
        else:
            loss_obj = loss.mean()

        del x, y, encoder_attention_mask, decoder_attention_mask

        # torch.cuda.empty_cache()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        loss_obj.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # validate
        # num_correct = sum(loss < min_loss) if min_loss is not None else 0

        self.model.eval()  # small hack but it's working
        acc = self.test_batch(batch)
        num_correct = acc.num_correct
        self.model.train()

        loss_item = loss_obj.item()

        if self.freeze_ratio > 0.0:
            if self.last_freeze_loss is None:
                        self.last_freeze_loss = loss_item
            elif loss <= 0.5 * self.last_freeze_loss:
                self.last_freeze_loss = loss_item
                print("Freezing half of the unfrozzen layers")
                self.freeze_remaining_layers()
                self.freeze_ratio = 0.5 * self.freeze_ratio + 0.5

        del loss

        return BatchResult(loss_item, num_correct)

    def train_batch_label_smoothing(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = batch

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
        # import pdb; pdb.set_trace()
        ret = torch.min(loss.view(n_labels,-1),dim=0)
        pred = ret.indices
        losses = ret.values

        # loss_obj = LabelSmoothingCrossEntropy(epsilon=self.epsilon)(-loss.view(-1,len(self.labels)),(correct_labels-min(self.labels)).to(self.device))

        l_idx = correct_labels - min(self.labels)           # get locations instead of tokens
        l_idx = l_idx.to(self.device)
        loss_obj = (1.0-self.epsilon) * loss.view(n_labels,-1).T.gather(1,l_idx.view(-1,1))  # smoothing 

        for i in range(1,n_labels):
            indices = (l_idx + i) % n_labels        # next location
            loss_obj += (self.epsilon / n_labels) * loss.view(n_labels,-1).T.gather(1,indices.view(-1,1))

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

    def test_batch(self, batch) -> BatchResult:
        x, encoder_attention_mask, y, decoder_attention_mask = batch
       
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
        with torch.no_grad():
            outputs = self.model(input_ids=inp_x, decoder_input_ids=inp_y, **model_kwargs)
            loss = outputs[0]
            loss = loss.view(inp_x.size(0), -1)
            loss = loss.mean(dim=1)
        # for i in range(len(self.labels)):
        #     for j in range(batch_size):
        #         curr_loss = loss[i * batch_size + j]
        #         if loss[i * batch_size + j] < best_res[j][0]:
        #             best_res[j] = (curr_loss, self.labels[i])

        ret = torch.min(loss.view(len(self.labels),-1),dim=0)
        pred = ret.indices
        losses = ret.values
                
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

        tot = total_loss.item()

        return BatchResult(tot, num_correct.item())


class DiscriminativeTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, max_len=128, num_labels=3, tokenizer=None, device=None):
        super().__init__(model, None, optimizer, scheduler, device)
        # self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.last_freeze_loss = None
        self.freeze_ratio = 0.0
        self.labels = torch.tensor(range(num_labels))

    def train_epoch(self, dl_train: DataLoader, **kw):
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        # import pdb; pdb.set_trace()
        if len(batch) == 3:         # H, P, y
            H,P,labels = batch
            input_dict = self.tokenizer.batch_encode_plus([[H[i],P[i]] for i in range(len(H))], pad_to_max_length=True, return_tensors='pt')
        elif len(batch) == 2:     #Hypotesis only
            H,labels = batch
            input_dict = self.tokenizer.batch_encode_plus(H, pad_to_max_length=True, return_tensors='pt')
        # import pdb; pdb.set_trace()
        batch_encoded = [input_dict[item] for item in ['input_ids', 'attention_mask']]
        x, attention_mask = batch_encoded
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # token_type_ids = token_type_ids.to(self.device)
        labels = labels.to(self.device)

        model_kwargs = {
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "labels": labels
        }

        if 'token_type_ids' in input_dict:
            token_type_ids = input_dict['token_type_ids']
            token_type_ids = token_type_ids.to(self.device)
            model_kwargs['token_type_ids'] = token_type_ids
        self.optimizer.zero_grad()

        outputs = self.model(input_ids=x, **model_kwargs)

        loss, logits = outputs[:2]
        loss = loss.mean()

        del x, attention_mask
        if 'token_type_ids' in input_dict:
            del token_type_ids

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # validate
        labels = labels.to('cpu')
        pred = torch.argmax(logits,dim=1).to('cpu')

        num_correct = torch.sum(labels==pred)

        loss_item = loss.item()

        del loss

        return BatchResult(loss_item, num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        if len(batch) == 3:         # H, P, y
            H,P,labels = batch
            input_dict = self.tokenizer.batch_encode_plus([[H[i],P[i]] for i in range(len(H))], pad_to_max_length=True, return_tensors='pt')
        elif len(batch) == 2:     #Hypotesis only
            H,labels = batch
            input_dict = self.tokenizer.batch_encode_plus(H, pad_to_max_length=True, return_tensors='pt')
        batch_encoded = [input_dict[item] for item in ['input_ids', 'attention_mask']]
        x, attention_mask = batch_encoded
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # token_type_ids = token_type_ids.to(self.device)
        labels = labels.to(self.device)

        model_kwargs = {
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
            "labels": labels
        }

        if 'token_type_ids' in input_dict:
            token_type_ids = input_dict['token_type_ids']
            token_type_ids = token_type_ids.to(self.device)
            model_kwargs['token_type_ids'] = token_type_ids
        
        with torch.no_grad():
            outputs = self.model(input_ids=x, **model_kwargs)

            loss, logits = outputs[:2]
            loss = loss.mean()

        del x, attention_mask
        if 'token_type_ids' in input_dict:
            del token_type_ids
        # import pdb; pdb.set_trace()
        # check accuracy
        labels = labels.to('cpu')
        pred = torch.argmax(logits,dim=1).to('cpu')

        num_correct = torch.sum(labels==pred)

        loss_item = loss.item()

        del loss
        return BatchResult(loss_item, num_correct.item())
