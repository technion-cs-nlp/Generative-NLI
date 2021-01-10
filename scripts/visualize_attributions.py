import torch, os, tqdm, sys
from src.data import DiscriminativeDataset
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict

if not os.path.isfile("stats.torch"):
    attribution_map = 'attributions_100_true'
    attribution_paths = [None] * 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir_prefix='./data/snli_1.0/cl_snli'
    hard_test_labels = []
    hard_test_lines = []
    test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')
    val_str = ('dev_matched' if 'mnli' in data_dir_prefix else 'val')

    files = [f for f in os.listdir(attribution_map) if os.path.isfile(os.path.join(attribution_map, f))]
    for i,prefix in enumerate(["train_set","val_set","test_set","hard_test_set"]):
        f = list(filter(lambda f: f.startswith(prefix), files))
        if len(f)>0:
            path_ = os.path.join(attribution_map, f[0])
            attribution_paths[i] = torch.load(path_, map_location=device)


    with open(data_dir_prefix + f'_{test_str}_lbl_file') as test_labels_file:
        test_labels = test_labels_file.readlines()
    with open(data_dir_prefix + f'_{test_str}_source_file') as test_lines_file:
        test_lines = test_lines_file.readlines()
    with open(data_dir_prefix + '_train_lbl_file') as train_labels_file:
        train_labels = train_labels_file.readlines()
    with open(data_dir_prefix + '_train_source_file') as train_lines_file:
        train_lines = train_lines_file.readlines()
    with open(data_dir_prefix + f'_{val_str}_lbl_file') as val_labels_file:
        val_labels = val_labels_file.readlines()
    with open(data_dir_prefix + f'_{val_str}_source_file') as val_lines_file:
        val_lines = val_lines_file.readlines()
    if os.path.isfile(data_dir_prefix + '_test_hard_lbl_file') and \
            os.path.isfile(data_dir_prefix + '_test_hard_source_file'):
        with open(data_dir_prefix + '_test_hard_lbl_file') as val_labels_file:
            hard_test_labels = val_labels_file.readlines()
        with open(data_dir_prefix + '_test_hard_source_file') as val_lines_file:
            hard_test_lines = val_lines_file.readlines()

    # data_args = {}
    # dataset = DiscriminativeDataset
    # tokenizer = None
    # max_len = 0
    tokenizer_attr = AutoTokenizer.from_pretrained('bert-base-uncased')

    res = {}

    for check_lines, name in zip([train_lines, val_lines, test_lines, hard_test_lines], ['train','val','test','hard_test']):
        res[name] = \
        {
            'count':defaultdict(int),
            'ratio':defaultdict(float),
            'count_normal':defaultdict(int),
            'ratio_normal':defaultdict(float),
        }
        num_lines = len(check_lines)
        with tqdm.tqdm(desc='Saving...', total=num_lines,
                    file=sys.stdout) as pbar:
            ind = ['train','val','test','hard_test'].index(name)
            attribution_map = attribution_paths[ind]
            for index,line in enumerate(check_lines):
                split = line.split('|||')
                premise = split[0]
                # premise_len = len(premise.split(' '))
                hypothesis = split[1].replace('\n', '')
                
                premise_encoded = tokenizer_attr(premise,return_tensors='pt').input_ids.view(-1)
                premise_len = len(premise_encoded)
                premise_attr = attribution_map[index].view(-1)[:premise_len]
                for threshold in np.arange(-1.0,1.1,0.1):
                    threshold = round(threshold,2)
                    # if threshold == -1.0:
                    #     # import pdb; pdb.set_trace()
                    #     pass
                    mask = premise_attr >= threshold
                    premise_encoded_filtered = premise_encoded[mask]
                    premise_filtered = tokenizer_attr.decode(premise_encoded_filtered,skip_special_tokens=True)
                    word_filtered = premise_len - len(premise_encoded_filtered)
                    ratio_filtered = float(word_filtered) / premise_len
                    res[name]['count'][threshold] += word_filtered / num_lines
                    res[name]['ratio'][threshold] += ratio_filtered / num_lines

                    premise_attr_normal = premise_attr / premise_attr.sum()
                    mask = premise_attr_normal >= threshold
                    premise_encoded_filtered = premise_encoded[mask]
                    premise_filtered = tokenizer_attr.decode(premise_encoded_filtered,skip_special_tokens=True)
                    word_filtered_normal = premise_len - len(premise_encoded_filtered)
                    ratio_filtered_normal = float(word_filtered_normal) / premise_len
                    res[name]['count_normal'][threshold] += word_filtered_normal / num_lines
                    res[name]['ratio_normal'][threshold] += ratio_filtered_normal / num_lines
        
                pbar.update()


    torch.save(res, "stats.torch")


import matplotlib.pyplot as plt
dic = torch.load("stats.torch")
fig, axs = plt.subplots(4, 2)

plt.setp(axs, xticks=list(dic['train']['ratio'].keys())[0:13])
for i,name in enumerate(['train','val','test','hard_test']):
    for j,(group,color) in enumerate(zip(['ratio','ratio_normal'],['tab:blue','tab:red'])):
        axs[i, j].plot(list(dic[name][group].keys())[0:13], list(dic[name][group].values())[0:13], color)
        axs[i, j].set_title(f'{name}-{group}')



for ax in axs.flat:
    ax.set(xlabel='Threshold', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

figure = plt.gcf()
figure.set_size_inches(10, 16)
plt.savefig('thres.png')