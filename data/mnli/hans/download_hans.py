import requests
import os
import numpy as np

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"

def ensure_dir_exists(filename):
    """Make sure the parent directory of `filename` exists"""
    os.makedirs(dirname(filename), exist_ok=True)

def download_to_file(url, output_file):
    """Download `url` to `output_file`, intended for small files."""
    ensure_dir_exists(output_file)
    with requests.get(url) as r:
        r.raise_for_status()
        with open(output_file, 'wb') as f:
            f.write(r.content)

def load_hans_subsets():
    src = "heuristics_evaluation_set.txt"
    if not exists(src):
        print("Downloading source to %s..." % src)
        utils.download_to_file(HANS_URL, src)

    hans_datasets = []
    labels = ["entailment", "non-entailment"]
    subsets = set()
    with open(src, "r") as f:
        for line in f.readlines()[1:]:
            line = line.split("\t")
            subsets.add(line[-3])
    subsets = [x for x in subsets]

    for label in labels:
        for subset in subsets:
            name = "hans_{}_{}".format(label, subset)
            examples = load_hans(filter_label=label, filter_subset=subset)
            hans_datasets.append((name, examples))

    return hans_datasets


def load_hans(n_samples=None, filter_label=None, filter_subset=None) -> List[
    TextPairExample]:
    out = []

    if filter_label is not None and filter_subset is not None:
        print("Loading hans subset: {}-{}...".format(filter_label, filter_subset))
    else:
        print("Loading hans all...")

    src = "heuristics_evaluation_set.txt"
    if not exists(src):
        print("Downloading source to %s..." % src)
        utils.download_to_file(HANS_URL, src)

    with open(src, "r") as f:
        f.readline()
        lines = f.readlines()

    if n_samples is not None:
        lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples,
                                                                replace=False)

    for line in lines:
        parts = line.split("\t")
        label = parts[0]

        if filter_label is not None and filter_subset is not None:
            if label != filter_label or parts[-3] != filter_subset:
                continue

        if label == "non-entailment":
            label = 0
        elif label == "entailment":
            label = 1
        else:
            raise RuntimeError()
        s1, s2, pair_id = parts[5:8]
        out.append(TextPairExample(pair_id, s1, s2, label))
    return out
