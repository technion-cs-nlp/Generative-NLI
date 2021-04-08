import os
from dataclasses import field, dataclass
from os.path import join
from typing import List, Tuple

from datasets import load_dataset, Dataset

_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DatasetInfo:
    dataset_config_file: str = None
    eval_datasets: List[str] = field(default_factory=lambda: [])
    ignored_keys: List[str] = field(default_factory=lambda: [])
    sentence1_key: str = 'premise'
    sentence2_key: str = 'hypothesis'
    train_dataset_name: str = 'train'
    dev_dataset_name: str = 'validation'
    test_dataset_name: str = 'test'
    binerize: bool = False


DATASETS = {
    'multi_nli': DatasetInfo(
        dataset_config_file='multi_nli.py',
        eval_datasets=['hans', 'multi_nli_hard_matched', 'multi_nli_hard_mismatched'],
        ignored_keys=['hypothesis_parse', 'premise_parse', 'id'],
        dev_dataset_name='validation_matched',
        test_dataset_name='validation_mismatched'
    ),
    'snli': DatasetInfo(
        eval_datasets=['hans', 'snli_hard']
    ),
    'fever_nli': DatasetInfo(
        dataset_config_file='fever_nli.py',
        eval_datasets=['fever_symmetric'],
        ignored_keys=['id'],
        sentence1_key='evidence',
        sentence2_key='claim',
        test_dataset_name='validation'
    ),
    'hans': DatasetInfo(
        test_dataset_name='validation',
        binerize=True
    ),
    'fever_symmetric': DatasetInfo(
        dataset_config_file='fever_symmetric.py',
        ignored_keys=['id'],
        sentence1_key='evidence',
        sentence2_key='claim',
        binerize=True
    ),
    'fever_symmetricv2': DatasetInfo(
        dataset_config_file='fever_symmetricv2.py',
        ignored_keys=['id'],
        sentence1_key='evidence',
        sentence2_key='claim',
        binerize=True
    ),
    'multi_nli_hard_matched': DatasetInfo(
        dataset_config_file='multi_nli_hard.py',
        test_dataset_name='test_matched'
    ),
    'multi_nli_hard_mismatched': DatasetInfo(
        dataset_config_file='multi_nli_hard.py',
        test_dataset_name='test_mismatched'
    ),
    'snli_hard': DatasetInfo(
        dataset_config_file='snli_hard.py'
    )
}


def load_dataset_aux(train_dataset_name: str) -> Tuple[Dataset, DatasetInfo]:
    if train_dataset_name in DATASETS:
        dataset_info: DatasetInfo = DATASETS[train_dataset_name]
        train_dataset_name = os.path.abspath(join(_DIR, dataset_info.dataset_config_file)) \
            if dataset_info.dataset_config_file is not None else train_dataset_name
    else:
        dataset_info = DatasetInfo()
    datasets = load_dataset(train_dataset_name)
    return datasets, dataset_info
