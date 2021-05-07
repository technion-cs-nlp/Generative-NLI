import os, torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

x = torch.load('likelihoods/hyp_only_likelihoods.torch', map_location=torch.device('cpu'))
y_disc = torch.load('likelihoods/disc_likelihoods.torch', map_location=torch.device('cpu'))
y_gen = torch.load('likelihoods/gen_likelihoods.torch', map_location=torch.device('cpu'))
y_rand = torch.load('likelihoods/rand_likelihoods.torch', map_location=torch.device('cpu'))

with open('data/snli_1.0/cl_snli_test_lbl_file') as f:
    test_labels = f.readlines()
with open('data/snli_1.0/cl_snli_train_lbl_file') as f:
    train_labels = f.readlines()

most_common = max(set(train_labels), key=train_labels.count)
# import pdb; pdb.set_trace()
y_maj = torch.tensor([(1.0 if sam==most_common else 0.0) for sam in test_labels])
bias_probs = (-x).exp()
rand_probs = (-y_rand).exp()
disc_probs = (-y_disc).exp()
gen_probs = (-y_gen).exp()
temp = 0.001
# scaled_bias_probs=bias_probs**(1/temp)
# bias_probs = scaled_bias_probs/scaled_bias_probs.sum()
fig, ax = plt.subplots(2,2,sharey=True,sharex=True)
ax[0,0].scatter(bias_probs, rand_probs, s=0.3, alpha=0.5)
ax[0,0].set_title("Random model",fontsize=13)
corr, _ = pearsonr(bias_probs,rand_probs)
print(f'Correlation with random: {corr}')

ax[0,1].scatter(bias_probs, y_maj, s=0.3, alpha=0.5)
ax[0,1].set_title('Majority model',fontsize=13)
corr, _ = pearsonr(bias_probs,y_maj)
print(f'Correlation with majority: {corr}')

ax[1,0].scatter(bias_probs, disc_probs, s=0.3, alpha=0.5)
ax[1,0].set_title('Discriminative model',fontsize=13)
corr, _ = pearsonr(bias_probs,disc_probs)
print(f'Correlation with discriminative: {corr}')
# scaled_disc_probs=disc_probs**(1/temp)
# probs = scaled_disc_probs/scaled_disc_probs.sum()
# corr, _ = pearsonr(probs, bias_probs)
# print(f'Correlation with discriminative: {corr}')

ax[1,1].scatter(bias_probs, gen_probs, s=0.3, alpha=0.5)
ax[1,1].set_title("Generative model",fontsize=13)
corr, _ = pearsonr(bias_probs, gen_probs)
print(f'Correlation with generative: {corr}')
# scaled_gen_probs=gen_probs**(1/temp)
# probs = scaled_gen_probs/scaled_gen_probs.sum()
# corr, _ = pearsonr(probs, bias_probs)
# print(f'Correlation with generative: {corr}')

fig.text(0.5, 0.03, 'p(y|H)', ha='center',fontsize=13)
fig.text(0.03, 0.5, 'p(y|P,H)', va='center', rotation='vertical', fontsize=13)
plt.rc('axes', labelsize=20)
plt.savefig('all.png',dpi=300)
plt.clf()
