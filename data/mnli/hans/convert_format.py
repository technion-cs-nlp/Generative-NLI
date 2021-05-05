

source_file = "heuristics_evaluation_set.txt"
heuristics = ['lexical_overlap', 'subsequence', 'constituent']
label_mapping = {'entailment': 'entailment', 'non-entailment': 'neutral'}

source_content = {h: [] for h in heuristics}
labels = {h: [] for h in heuristics}
ids = {h: [] for h in heuristics}

with open(source_file, 'r') as f:
    header = next(f)
    for line in f:
        parts = line.strip().split('\t')
        label = parts[0]
        premise = parts[5]
        hypothesis = parts[6]
        heuristic = parts[8]
        
        source_i = f'{premise}|||{hypothesis}'
        label_i = label_mapping[label]
        id_i = parts[7][2:]

        source_content[heuristic].append(source_i)
        labels[heuristic].append(label_i)
        ids[heuristic].append(id_i)

for h in heuristics:
    source_fname = f'hans_evalset_{h}_source_file'
    with open(source_fname, 'w') as f:
        for source_i in source_content[h]:
            f.write(source_i+'\n')

    label_fname = f'hans_evalset_{h}_lbl_file'
    with open(label_fname, 'w') as f:
        for label_i in labels[h]:
            f.write(label_i+'\n')

    id_fname = f'hans_evalset_{h}_id_file'
    with open(id_fname, 'w') as f:
        for id_i in ids[h]:
            f.write(id_i+'\n')
