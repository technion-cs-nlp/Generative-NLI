from src.models import get_model
from src.data import HypothesisOnlyDataset
from transformers import AutoTokenizer
import torch, sys, tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model_path = "checkpoints/exp_bart_b_hyp_mnli_model"
model_path = "checkpoints/overlap_200_hyp_model"
#model_name = "facebook/bart-base"
model_name = "facebook/bart-base"
data_path = "data/mnli/hans/train/cl_multinli_dev_mismatched"
source_path = f"{data_path}_source_file"
lbl_path = f"{data_path}_lbl_file"
hard_source_path = f"{data_path}_hard_source_file"
hard_lbl_path = f"{data_path}_hard_lbl_file"

with open(lbl_path) as labels_file:
    labels = labels_file.readlines()
with open(source_path) as lines_file:
    lines = lines_file.readlines()

dataset = HypothesisOnlyDataset(lines,labels)

tokenizer = AutoTokenizer.from_pretrained(model_name if 'patrick' not in model_name else 'bert-base-uncased')
tokenizer_decoder = None

model = get_model(tokenizer=tokenizer,
                      tokenizer_decoder=tokenizer_decoder,
                      model='disc',
                      model_name=model_name,
                      decoder_model_name=None,
                      model_path=model_path)
model = model.to(device)

def _prepare_batch(batch):
        input_dict = {}
        labels = None
        if len(batch) == 3:  # P, H, y
            P, H, labels = batch[0:3][0:3]
            input_dict = tokenizer.batch_encode_plus([[P[i], H[i]] for i in range(len(P))], padding='longest',
                                                          return_tensors='pt')
        elif len(batch) == 2:  # Hypotesis only
            H, labels = batch
            H = list(H)
            input_dict = tokenizer.batch_encode_plus(H, padding='longest', return_tensors='pt')

        batch_encoded = [input_dict[item].to(device) for item in ['input_ids', 'attention_mask']]
        batch_encoded += [labels.to(device)]

        if 'token_type_ids' in input_dict:
            token_type_ids = input_dict['token_type_ids']
            batch_encoded += [token_type_ids.to(device)]
        else:
            batch_encoded += [None]

        return batch_encoded

_sum = 0
possible_labels = ['contradiction', 'entailment', 'neutral']
with tqdm.tqdm(desc='Saving...', total=len(dataset),
                    file=sys.stdout) as pbar:
    with open(hard_source_path,'w') as hard_source_file:
        with open(hard_lbl_path, 'w') as hard_lbl_file:
            for line,l in zip(lines, labels):
                line, l = line.strip(), l.strip()
                P,H = line.split('|||')[0:2]
                y = possible_labels.index(l)
                y = torch.tensor(y)
                batch = ([H],y)
                x, attention_mask, labels, token_type_ids = _prepare_batch(batch)
                x = x.to(device)
                attention_mask = attention_mask.to(device)
                # token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)

                model_kwargs = {
                    "attention_mask": attention_mask,
                    # "token_type_ids": token_type_ids,
                    "labels": labels
                }

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                    model_kwargs['token_type_ids'] = token_type_ids

                with torch.no_grad():
                    outputs = model(input_ids=x, **model_kwargs)

                    loss, logits = outputs[:2]
                    loss = loss.mean()

                del x, attention_mask
                if token_type_ids is not None:
                    del token_type_ids

                # check accuracy
                labels = labels.to('cpu')
                pred = torch.argmax(logits, dim=1).to('cpu')
                # import pdb; pdb.set_trace()
                if labels != pred[0]:
                    hard_source_file.write(f"{P}|||{H}\n")
                    hard_lbl_file.write(f"{l}\n")
                else: 
                    _sum +=1
                pbar.update()

print(f'accuracy={float(_sum)/len(lines) * 100}')
