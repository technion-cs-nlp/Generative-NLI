from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np

hard = True
hard_str = ''
if hard:
    hard_str='_hard'

with open('loss_results/bart_cont'+hard_str) as f:
    cont_lines = f.readlines()

with open('loss_results/bart_ent'+hard_str) as f:
    ent_lines = f.readlines()

with open('loss_results/bart_neut'+hard_str) as f:
    neut_lines = f.readlines()

with open('/home/dimion/PremiseGeneratorBert/data/snli_1.0/cl_snli_test'+hard_str+'_lbl_file') as f:
    labels = f.readlines()
normal = [1,1,1]      ## regular
# normal = [61.238,67.922,49.642]
# normal = [49.909,50.972,41.950]
# normal = [0.32949918566775244, 0.34283387622149836, 0.3276669381107492]     ## percent in test
normal = [2.0/3-i/sum(normal) for i in normal]
s=0
loss_cont=defaultdict(float)
loss_ent=defaultdict(float)
loss_neut=defaultdict(float)
pred = []
true = []
for c,e,n,l in zip(cont_lines,ent_lines,neut_lines,labels):
    c = float(c.strip()) * normal[0]
    e = float(e.strip()) * normal[1]
    n = float(n.strip()) * normal[2]
    l = (l.strip())
    true.append(l)
    loss_cont[l]+=c
    loss_ent[l]+=e
    loss_neut[l]+=n
    if c <= min(c,e,n):
        if l == 'contradiction':
            s+=1
        pred.append('contradiction')
    elif e <= min(c,e,n):
        if l == 'entailment':
            s+=1
        pred.append('entailment')
    else:
        if l == 'neutral':
            s+=1
        pred.append('neutral')
labels=["contradiction", "entailment","neutral"]
for d in (loss_cont, loss_ent, loss_neut):
    for k,v in d.items():
        d[k] = v / len([l for l in true if l==k])
cm = confusion_matrix(true, pred, labels=labels)
np.save('one_label_model.res',pred)
import pdb; pdb.set_trace()
print(f'Accuracy: {s/len(cont_lines)*100}%')