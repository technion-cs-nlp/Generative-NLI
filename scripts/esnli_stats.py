import os
import pandas as pd

esnli = pd.read_csv('data/esnli/esnli_test.csv')
marked_premises = esnli['Sentence1_marked_1']

def check_mark(word: str):
    return word.startswith('*') and word.endswith('*')

stats = {
    'count':0,
    'ratio':0.0,
    'word count': 0
}

for line in marked_premises:
    splited = line.split()
    stats['word count'] += len(splited)
    if '*' not in line:
        # stats['count'] += len(splited)
        # stats['ratio'] += 1.0
        stats['count'] += 0
        stats['ratio'] += 0.0
    else:
        highlighted = [word for word in splited if check_mark(word)]
        stats['count'] += len(highlighted)
        stats['ratio'] += float(len(highlighted)) / len(splited)

print('No highlight are with 0 "important words"')
print(f'Average important words per premise: {float(stats["count"])/len(marked_premises)}')
print(f'Average ratio of important words per premise: {float(stats["ratio"])/len(marked_premises)}')
print(f'Average ratio of important words in test set: {float(stats["count"])/stats["word count"]}')

stats = {
    'count':0,
    'ratio':0.0,
    'word count': 0
}
no_highlight = 0

for line in marked_premises:
    splited = line.split()
    if '*' not in line:
        no_highlight += 1
        continue
    else:
        stats['word count'] += len(splited)
        highlighted = [word for word in splited if check_mark(word)]
        stats['count'] += len(highlighted)
        stats['ratio'] += float(len(highlighted)) / len(splited)

print('No highlight are filterd out')
print(f'Average important words per premise: {float(stats["count"])/(len(marked_premises)-no_highlight)}')
print(f'Average ratio of important words per premise: {float(stats["ratio"])/(len(marked_premises)-no_highlight)}')
print(f'Average ratio of important words in test set: {float(stats["count"])/stats["word count"]}')
