# -*- coding:utf-8 _*-

import numpy as np
from scipy.stats import pearsonr
from kendalltau import compute_kenall

# set use classifier
use_cls = False

system_5 = ['BART', 'IBT', 'NIU']
system_4 = ['BART', 'IBT', 'NIU', 'HIGH']
system_3 = ['BART', 'IBT', 'NIU', 'HIGH', 'RAO']
system_2 = ['BART', 'IBT', 'NIU', 'HIGH', 'RAO', 'YI']
system_1 = ['BART', 'IBT', 'NIU', 'HIGH', 'RAO', 'YI', 'ZHOU']
system_0 = ['BART', 'IBT', 'NIU', 'HIGH', 'RAO', 'YI', 'ZHOU', 'LUO']
# system_1 = ['LUO', 'ZHOU', 'YI', 'RAO', 'HIGH', 'NIU', 'IBT']
# system_2 = ['LUO', 'ZHOU', 'YI', 'RAO', 'HIGH', 'NIU']
# system_3 = ['LUO', 'ZHOU', 'YI', 'RAO', 'HIGH']
# system_4 = ['LUO', 'ZHOU', 'YI', 'RAO']
# system_5 = ['LUO', 'ZHOU', 'YI']

for systems in [system_0, system_1, system_2, system_3, system_4, system_5]:
    human, auto = [], []

    # segment-level
    for s in systems:
        temp_0 = []
        temp_1 = []
        with open('data/outputs/{}.reg.pt16.txt'.format(s), 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split('\t')
                temp_0.append((float(line[5]) + float(line[8])) / 2)

                # for style classifier
                if not use_cls:
                    temp_1.append(float(line[-1]))
                else:
                    if i < 40:
                        temp_1.append(float(line[-1]))
                    else:
                        temp_1.append(1-float(line[-1]))
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
        with open('data/outputs/{}.reg.pt16.txt'.format(s), 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split('\t')
                temp_0.append((float(line[5]) + float(line[8])) / 2)
                temp_1.append(float(line[-1]))

                # for style classifier
                if not use_cls:
                    temp_1.append(float(line[-1]))
                else:
                    if i < 40:
                        temp_1.append(float(line[-1]))
                    else:
                        temp_1.append(1 - float(line[-1]))
        human.append(np.mean(temp_0))
        auto.append(np.mean(temp_1))
    print('Pearson (System):', pearsonr(human, auto))
