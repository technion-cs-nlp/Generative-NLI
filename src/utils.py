from typing import NamedTuple, List


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


def get_max_len(l, sep, tokenizer):
    max_len = 0
    it = 0
    for sent in l:
        # print(it)
        it += 1
        split = sent.split(sep)
        max_len = max([max_len, len(tokenizer.encode(split[0])), len(tokenizer.encode(split[1])) + 1])

    return max_len
