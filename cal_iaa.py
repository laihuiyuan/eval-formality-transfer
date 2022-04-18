# -*- coding:utf-8 _*-

import numpy as np
from scipy.stats import pearsonr

anno = [[] for i in range(4)]
systems = ['BART', 'HIGH', 'IBT', 'LUO', 'NIU', 'RAO', 'YI', 'ZHOU']

for s in systems:
    with open('data/outputs/{}.human.txt'.format(s), 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            if int(line[0])==1:
                anno[0].append([float(i) for i in line[4:10]])
            elif int(line[0])==2:
                anno[1].append([float(i) for i in line[4:10]])
            elif int(line[0]) == 4:
                anno[2].append([float(i) for i in line[4:10]])
            else:
                anno[3].append([float(i) for i in line[4:10]])

anno = np.array(anno)
aspects = ['style: ','content: ','fluency: ']
for i in range(4):
    print('---Survey: {}---'.format(i))
    temp_0, temp_1 = [], []
    for j in range(3):
        temp_0.extend(list(anno[i,:,j]))
        temp_1.extend(list(anno[i,:,j+3]))
        print(aspects[j], pearsonr(anno[i,:,j],anno[i,:,j+3]))
    print('overall:', pearsonr(temp_0, temp_1))
anno = anno.reshape(640, 6)
print('Overall:')
for i in range(3):
    print(pearsonr(anno[:,i], anno[:,i+3]))
