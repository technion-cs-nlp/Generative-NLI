from src.data_miki.datasets import load_dataset_aux
fever, _ = load_dataset_aux('fever_nli')
train_lines = fever['train']
train_labels = None
val_lines = fever['validation']
val_labels = None
fever_test, _ = load_dataset_aux('fever_symmetric')
test_lines = fever_test['test']
test_labels = None
fever_testv2, _ = load_dataset_aux('fever_symmetricv2')
hard_test_lines = fever_testv2['test']
hard_test_labels = None
labels = ["SUPPORTS", "REFUTES", "NOT-ENOUGH-INFO"]
ids = []
# import pdb;pdb.set_trace()
for set_, name in zip([train_lines, val_lines, test_lines, hard_test_lines],['train', 'val', 'test', 'test_hard']):
	if name != 'train':
		continue
	with open(f'data/fever_temp/fever_{name}_source_file','w') as f_source:
		with open(f'data/fever_temp/fever_{name}_lbl_file','w') as f_lbl:
			for sam in set_:
				# import pdb; pdb.set_trace()
				# if sam["label"] != 2:
				if sam['id'] in ids:
					continue
				ids.append(sam['id'])
				f_source.write(f'{sam["evidence"]}|||{sam["claim"]}\n')
				f_lbl.write(f'{labels[sam["label"]]}\n')