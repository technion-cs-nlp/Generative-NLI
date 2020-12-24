import torch, os, tqdm, sys
from src.data import DiscriminativeDataset
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict

attribution_map = 'attributions'
attribution_paths = [None] * 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir_prefix='./data/snli_1.0/cl_snli'
hard_test_labels = []
hard_test_lines = []
test_str = ('dev_mismatched' if 'mnli' in data_dir_prefix else 'test')
val_str = ('dev_matched' if 'mnli' in data_dir_prefix else 'val')

files = [f for f in os.listdir(attribution_map) if os.path.isfile(os.path.join(attribution_map, f))]
for i,prefix in enumerate(["train_set","validation_set","test_set","hard_test_set"]):
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

res = \
    {
        'count':defaultdict(int),
        'ratio':defaultdict(float),
        'count_normal':defaultdict(int),
        'ratio_normal':defaultdict(float),
    }
num_lines = len(train_lines)

with tqdm.tqdm(desc='Saving...', total=num_lines,
               file=sys.stdout) as pbar:
    attribution_map = attribution_paths[0]
    for index,line in enumerate(train_lines):
        split = line.split('|||')
        premise = split[0]
        # premise_len = len(premise.split(' '))
        hypothesis = split[1].replace('\n', '')
        
        premise_encoded = tokenizer_attr(premise,return_tensors='pt').input_ids.view(-1)
        premise_len = len(premise_encoded)
        premise_attr = attribution_map[index].view(-1)[:premise_len]
        for threshold in np.arange(0.0,-1.1,-0.1):
            if threshold == -1.0:
                # import pdb; pdb.set_trace()
                pass
            mask = premise_attr >= threshold
            premise_encoded_filtered = premise_encoded[mask]
            premise_filtered = tokenizer_attr.decode(premise_encoded_filtered,skip_special_tokens=True)
            word_filtered = premise_len - len(premise_encoded_filtered)
            ratio_filtered = float(word_filtered) / premise_len
            res['count'][threshold] += word_filtered / num_lines
            res['ratio'][threshold] += ratio_filtered / num_lines

            premise_attr_normal = premise_attr / premise_attr.sum()
            mask = premise_attr_normal >= threshold
            premise_encoded_filtered = premise_encoded[mask]
            premise_filtered = tokenizer_attr.decode(premise_encoded_filtered,skip_special_tokens=True)
            word_filtered_normal = premise_len - len(premise_encoded_filtered)
            ratio_filtered_normal = float(word_filtered_normal) / premise_len
            res['count_normal'][threshold] += word_filtered_normal / num_lines
            res['ratio_normal'][threshold] += ratio_filtered_normal / num_lines
    
        pbar.update()


torch.save(res, "stats.torch")
# if attribution_map is not None:
#     data_args['attribution_map'] = attribution_paths[2]
# ds_test = dataset(test_lines, test_labels, tokenizer, max_len=max_len, **data_args)

# if attribution_map is not None:
#     data_args['attribution_map'] = attribution_paths[1]
# ds_val = dataset(val_lines, val_labels, tokenizer, max_len=max_len, **data_args)

# if attribution_map is not None:
#     data_args['attribution_map'] = attribution_paths[0]
# ds_train = dataset(train_lines, train_labels, tokenizer, max_len=max_len, **data_args)

# if attribution_map is not None:
#     data_args['attribution_map'] = attribution_paths[3]
# else:
#     data_args.pop('attribution_map',None)
# ds_hard_test = dataset(hard_test_lines, hard_test_labels, tokenizer, max_len=max_len, **data_args)

pass