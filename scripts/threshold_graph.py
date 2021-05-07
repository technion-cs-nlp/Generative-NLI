import torch
import numpy as np
import matplotlib.pyplot as plt

dic = torch.load("stats.torch")
temp = dic['test']['ratio']
import pdb; pdb.set_trace()
temp = [v for k,v in temp.items()][:21]
# temp = [1-temp[0]] + 
temp = [(temp[i]-temp[i-1]) for i in range(1,len(temp))]

def stringify(x):
    if x < 0.0:
        return f'({x})'
    else:
        return f'{np.abs(x)}'

ticks = [f'{stringify(round(i-0.1,2))}-{stringify(round(i,2))}' for i in np.arange(-0.9,1.1,0.1)]
# ticks = ['<0.0']+ticks

f, ax = plt.subplots(figsize=(8,8))
# import pdb; pdb.set_trace()
plt.bar(ticks,temp)
plt.xticks(rotation=90)
plt.ylabel("Average amount of words from each premise")
plt.xlabel("Attribution range")

plt.savefig("count_hist.png")