import os, torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

x_01 = torch.load('likelihoods/hyp_01_hard.torch', map_location=torch.device('cpu'))
y_disc01 = torch.load('likelihoods/disc_01_hard.torch', map_location=torch.device('cpu'))
y_gen01 = torch.load('likelihoods/gen_01_hard.torch', map_location=torch.device('cpu'))
x_03 = torch.load('likelihoods/hyp_03_hard.torch', map_location=torch.device('cpu'))
y_disc03 = torch.load('likelihoods/disc_03_hard.torch', map_location=torch.device('cpu'))
y_gen03 = torch.load('likelihoods/gen_03_hard.torch', map_location=torch.device('cpu'))
x_05 = torch.load('likelihoods/hyp_05_hard.torch', map_location=torch.device('cpu'))
y_disc05 = torch.load('likelihoods/disc_05_hard.torch', map_location=torch.device('cpu'))
y_gen05 = torch.load('likelihoods/gen_05_hard.torch', map_location=torch.device('cpu'))
x_07 = torch.load('likelihoods/hyp_07_hard.torch', map_location=torch.device('cpu'))
y_disc07 = torch.load('likelihoods/disc_07_hard.torch', map_location=torch.device('cpu'))
y_gen07 = torch.load('likelihoods/gen_07_hard.torch', map_location=torch.device('cpu'))
x_09 = torch.load('likelihoods/hyp_09_hard.torch', map_location=torch.device('cpu'))
y_disc09 = torch.load('likelihoods/disc_09_hard.torch', map_location=torch.device('cpu'))
y_gen09 = torch.load('likelihoods/gen_019_hard.torch', map_location=torch.device('cpu'))

x_01_unbiased = torch.load('likelihoods/hyp_unbiased_hard.torch', map_location=torch.device('cpu'))
y_disc01_unbiased = torch.load('likelihoods/disc_01_unbiased_hard.torch', map_location=torch.device('cpu'))
y_gen01_unbiased = torch.load('likelihoods/gen_01_unbiased_hard.torch', map_location=torch.device('cpu'))
x_03_unbiased = torch.load('likelihoods/hyp_unbiased_hard.torch', map_location=torch.device('cpu'))
y_disc03_unbiased = torch.load('likelihoods/disc_03_unbiased_hard.torch', map_location=torch.device('cpu'))
y_gen03_unbiased = torch.load('likelihoods/gen_03_unbiased_hard.torch', map_location=torch.device('cpu'))
x_05_unbiased = torch.load('likelihoods/hyp_unbiased_hard.torch', map_location=torch.device('cpu'))
y_disc05_unbiased = torch.load('likelihoods/disc_05_unbiased_hard.torch', map_location=torch.device('cpu'))
y_gen05_unbiased = torch.load('likelihoods/gen_05_unbiased_hard.torch', map_location=torch.device('cpu'))
x_07_unbiased = torch.load('likelihoods/hyp_unbiased_hard.torch', map_location=torch.device('cpu'))
y_disc07_unbiased = torch.load('likelihoods/disc_07_unbiased_hard.torch', map_location=torch.device('cpu'))
y_gen07_unbiased = torch.load('likelihoods/gen_07_unbiased_hard.torch', map_location=torch.device('cpu'))
x_09_unbiased = torch.load('likelihoods/hyp_unbiased_hard.torch', map_location=torch.device('cpu'))
y_disc09_unbiased = torch.load('likelihoods/disc_09_unbiased_hard.torch', map_location=torch.device('cpu'))
y_gen09_unbiased = torch.load('likelihoods/gen_09_unbiased_hard.torch', map_location=torch.device('cpu'))

y_rand = torch.load('likelihoods/rand_hard.torch', map_location=torch.device('cpu'))
# y_ft = torch.load('likelihoods/bart_ft.torch', map_location=torch.device('cpu'))

with open('data/mnli/cl_multinli_dev_mismatched_hard_lbl_file') as f:
    test_labels = f.readlines()
with open('data/mnli/cl_multinli_train_lbl_file') as f:
    train_labels = f.readlines()

most_common = max(set(train_labels), key=train_labels.count)
# import pdb; pdb.set_trace()
y_maj = torch.tensor([(1.0 if sam==most_common else 0.0) for sam in test_labels])
bias_probs01 = (-x_01).exp()
bias_probs03 = (-x_03).exp()
bias_probs05 = (-x_05).exp()
bias_probs07 = (-x_07).exp()
bias_probs09 = (-x_09).exp()
rand_probs = (-y_rand).exp()
disc_probs01 = (-y_disc01).exp()
disc_probs03 = (-y_disc03).exp()
disc_probs05 = (-y_disc05).exp()
disc_probs07 = (-y_disc07).exp()
disc_probs09 = (-y_disc09).exp()
gen_probs01 = (-y_gen01).exp()
gen_probs03 = (-y_gen03).exp()
gen_probs05 = (-y_gen05).exp()
gen_probs07 = (-y_gen07).exp()
gen_probs09 = (-y_gen09).exp()

bias_probs01_unbiased = (-x_01_unbiased).exp()
bias_probs03_unbiased = (-x_03_unbiased).exp()
bias_probs05_unbiased = (-x_05_unbiased).exp()
bias_probs07_unbiased = (-x_07_unbiased).exp()
bias_probs09_unbiased = (-x_09_unbiased).exp()
rand_probs = (-y_rand).exp()
disc_probs01_unbiased = (-y_disc01_unbiased).exp()
disc_probs03_unbiased = (-y_disc03_unbiased).exp()
disc_probs05_unbiased = (-y_disc05_unbiased).exp()
disc_probs07_unbiased = (-y_disc07_unbiased).exp()
disc_probs09_unbiased = (-y_disc09_unbiased).exp()
gen_probs01_unbiased = (-y_gen01_unbiased).exp()
gen_probs03_unbiased = (-y_gen03_unbiased).exp()
gen_probs05_unbiased = (-y_gen05_unbiased).exp()
gen_probs07_unbiased = (-y_gen07_unbiased).exp()
gen_probs09_unbiased = (-y_gen09_unbiased).exp()

print("Biased")
corr, _ = pearsonr(bias_probs01, disc_probs01)
print(f'Correlation with discriminative (p=0.1): {corr}')
corr, _ = pearsonr(bias_probs01, gen_probs01)
print(f'Correlation with generative (p=0.1): {corr}')

corr, _ = pearsonr(bias_probs03, disc_probs03)
print(f'Correlation with discriminative (p=0.3): {corr}')
corr, _ = pearsonr(bias_probs03, gen_probs03)
print(f'Correlation with generative (p=0.3): {corr}')

corr, _ = pearsonr(bias_probs05, disc_probs05)
print(f'Correlation with discriminative (p=0.5): {corr}')
corr, _ = pearsonr(bias_probs05, gen_probs05)
print(f'Correlation with generative (p=0.5): {corr}')

corr, _ = pearsonr(bias_probs07, disc_probs07)
print(f'Correlation with discriminative (p=0.7): {corr}')
corr, _ = pearsonr(bias_probs07, gen_probs07)
print(f'Correlation with generative (p=0.7): {corr}')

corr, _ = pearsonr(bias_probs09, disc_probs09)
print(f'Correlation with discriminative (p=0.9): {corr}')
corr, _ = pearsonr(bias_probs09, gen_probs09)
print(f'Correlation with generative (p=0.9): {corr}')

print("Unbiased")
corr, _ = pearsonr(bias_probs01_unbiased, disc_probs01_unbiased)
print(f'Correlation with discriminative (p=0.1): {corr}')
corr, _ = pearsonr(bias_probs01_unbiased, gen_probs01_unbiased)
print(f'Correlation with generative (p=0.1): {corr}')

corr, _ = pearsonr(bias_probs03_unbiased, disc_probs03_unbiased)
print(f'Correlation with discriminative (p=0.3): {corr}')
corr, _ = pearsonr(bias_probs03_unbiased, gen_probs03_unbiased)
print(f'Correlation with generative (p=0.3): {corr}')

corr, _ = pearsonr(bias_probs05_unbiased, disc_probs05_unbiased)
print(f'Correlation with discriminative (p=0.5): {corr}')
corr, _ = pearsonr(bias_probs05_unbiased, gen_probs05_unbiased)
print(f'Correlation with generative (p=0.5): {corr}')

corr, _ = pearsonr(bias_probs07_unbiased, disc_probs07_unbiased)
print(f'Correlation with discriminative (p=0.7): {corr}')
corr, _ = pearsonr(bias_probs07_unbiased, gen_probs07_unbiased)
print(f'Correlation with generative (p=0.7): {corr}')

corr, _ = pearsonr(bias_probs09_unbiased, disc_probs09_unbiased)
print(f'Correlation with discriminative (p=0.9): {corr}')
corr, _ = pearsonr(bias_probs09_unbiased, gen_probs09_unbiased)
print(f'Correlation with generative (p=0.9): {corr}')