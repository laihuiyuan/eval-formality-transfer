# -*- coding:utf-8 _*-

import numpy as np
from scipy.stats import pearsonr
from kendalltau import compute_kenall

# systems = ['BART', 'HIGH', 'IBT', 'LUO', 'NIU', 'RAO', 'YI', 'ZHOU', 'REF']

system_5 = ['HIGH', 'NIU','BART']
system_4 = ['HIGH', 'NIU','BART','IBT']
system_3 = ['HIGH', 'NIU','BART','IBT','RAO']
system_2 = ['HIGH', 'NIU','BART','IBT','RAO','ZHOU']
system_1 = ['HIGH', 'NIU','BART','IBT','RAO','ZHOU','YI']
system_0 = ['HIGH', 'NIU','BART','IBT','RAO','ZHOU','YI','LUO']
# system_1 = ['NIU','BART','IBT','RAO','ZHOU','YI','LUO']
# system_2 = ['BART','IBT','RAO','ZHOU','YI','LUO']
# system_3 = ['IBT','RAO','ZHOU','YI','LUO']
# system_4 = ['RAO','ZHOU','YI','LUO']
# system_5 = ['ZHOU','YI','LUO']


for systems in [system_0, system_1, system_2, system_3, system_4, system_5]:
    human, auto = [], []

    # segment-level
    for s in systems:
        temp_0 = []
        temp_1 = []
        with open('data/outputs/{}.content.ref.txt'.format(s), 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split('\t')
                temp_0.append((float(line[4]) + float(line[7])) / 2)
                temp_1.append(float(line[-11]))
        human.append(temp_0)
        auto.append(temp_1)

    human = np.array(human)
    auto = np.array(auto)
    corr = []
    for i in range(human.shape[1]):
        try:
            corr.append(compute_kenall(human[:, i], auto[:, i]))
        except:
            continue
    print('-----')
    print('Kendalltau (Segment):', np.mean(corr))
    print('Pearson (Segment):', pearsonr(human.reshape(human.size),
                                         auto.reshape(auto.size)))

    # system-level
    human, auto = [], []
    for s in systems:
        temp_0, temp_1 = [], []
        with open('data/outputs/{}.content.ref.txt'.format(s), 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split('\t')
                temp_0.append((float(line[4]) + float(line[7])) / 2)
                temp_1.append(float(line[-11]))
        human.append(np.mean(temp_0))
        auto.append(np.mean(temp_1))
    print('Pearson (System):', pearsonr(human, auto))

