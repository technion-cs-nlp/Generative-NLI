import nltk
from scripts.self_bleu import  SelfBleu

# with open('data/self_bleu_data/snli_premises') as f:
#     hyp = f.readlines()
# with open('data/self_bleu_data/snli_bart_premises') as f:
#     ref = f.readlines()

bleu = SelfBleu('data/self_bleu_data/snli_premises')
hyp = bleu.get_reference()
bleu = SelfBleu('data/self_bleu_data/snli_bart_premises')
ref = bleu.get_reference()
references = [[r] for r in ref]

score = nltk.translate.bleu_score.corpus_bleu(references, hyp)

print(f'Bart generation quality based on the corresponding premise: {score*100}')

bleu = SelfBleu('data/self_bleu_data/snli_premises')
hyp = bleu.get_reference()
bleu = SelfBleu('data/self_bleu_data/snli_bart_ft_premises')
ref = bleu.get_reference()
references = [[r] for r in ref]
# import pdb; pdb.set_trace()
score = nltk.translate.bleu_score.corpus_bleu(references, hyp)

print(f'Bart fine-tuned generation quality based on the corresponding premise: {score*100}')

references = SelfBleu('data/self_bleu_data/snli_premises_filtered').get_reference()
# references = [list(x) for x in set(tuple(x) for x in references)]
corpus_bleu = SelfBleu('data/self_bleu_data/snli_bart_premises')
corpus_bleu.reference = references
# import pdb;pdb.set_trace()
score = corpus_bleu.get_score(is_fast=False)

print(f'Bart generation quality based on the entire dataset: {score*100}')

corpus_bleu = SelfBleu('data/self_bleu_data/snli_bart_ft_premises')
corpus_bleu.reference = references

score = corpus_bleu.get_score(is_fast=False)

print(f'Bart fine-tuned generation quality based on the entire dataset: {score*100}')