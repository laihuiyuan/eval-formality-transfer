# -*- coding:utf-8 _*-

import numpy as np
from scipy.stats import pearsonr
from kendalltau import compute_kenall

systems = ['BART', 'HIGH', 'IBT', 'LUO', 'NIU', 'RAO', 'YI', 'ZHOU']

# set a transfer direction
inf2for = True
for2inf = False


#segment-level
human, auto = [], []
for s in systems:
    temp_0 = []
    temp_1 = []
    with open('data/outputs/{}.ppl.txt'.format(s), 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')

            # for two different transfer directions
            if for2inf and i < 40:
                continue
            if inf2for and i >= 40:
                continue
            temp_0.append((float(line[6]) + float(line[9])) / 2)
            temp_1.append(float(line[-1]))
    human.append(temp_0)
    auto.append(temp_1)

human = np.array(human)
auto = np.array(auto)
corr = []
for i in range(human.shape[1]):
    try:
        corr.append(compute_kenall(human[:,i], auto[:,i]))
    except:
        continue
print('Kendalltau (Segment):', np.mean(corr))
print('Pearson (Segment):', pearsonr(human.reshape(320).tolist(),
                                     auto.reshape(320).tolist()))

# system-level
human, auto = [], []
for s in systems:
    temp_0, temp_1 = [], []
    with open('data/outputs/{}.ppl.txt'.format(s), 'r') as f:
        for i, line in enumerate(f.readlines()):
            # for two different transfer directions
            if for2inf and i < 40:
                continue
            if inf2for and i >= 40:
                continue
            line = line.strip().split('\t')
            temp_0.append((float(line[6]) + float(line[9])) / 2)
            temp_1.append(float(line[-1]))
    human.append(np.mean(temp_0))
    auto.append(np.mean(temp_1))
print('Pearson (System):', pearsonr(human,auto))