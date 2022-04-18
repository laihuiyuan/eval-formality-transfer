# -*- coding:utf-8 _*-

import numpy as np
from scipy import stats

systems = ['BART', 'HIGH', 'IBT', 'LUO', 'NIU', 'RAO', 'YI', 'ZHOU', 'REF']

survey_0, survey_1, survey_2, survey_3 = [], [], [], []

for s in systems:
    with open('data/outputs/{}.human.txt'.format(s), 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if int(line[0])==1:
                survey_0.append([float(i) for i in line[4:]])
            elif int(line[0])==2:
                survey_1.append([float(i) for i in line[4:]])
            elif int(line[0]) == 4:
                survey_2.append([float(i) for i in line[4:]])
            else:
                survey_3.append([float(i) for i in line[4:]])

survey_0 = stats.zscore(np.array(survey_0), axis=0)
survey_1 = stats.zscore(np.array(survey_1), axis=0)
survey_2 = stats.zscore(np.array(survey_2), axis=0)
survey_3 = stats.zscore(np.array(survey_3), axis=0)

for i, s in enumerate(systems):
    j = i * 2
    f0 = open('data/outputs/{}.human.norm.txt'.format(s), 'r')
    f1 = open('data/std/{}.txt'.format(s), 'w')
    data = survey_0[10*j:10*(j+1)].tolist()+survey_1[10*j:10*(j+1)].tolist()+\
           survey_2[10*j:10*(j+1)].tolist()+survey_3[10*j:10*(j+1)].tolist()+\
           survey_0[10*(j+1):10*(j+2)].tolist()+survey_1[10*(j+1):10*(j+2)].tolist()+\
           survey_2[10*(j+1):10*(j+2)].tolist()+survey_3[10*(j+1):10*(j+2)].tolist()
    for l0,l1 in zip(f0.readlines(), data):
        line = l0.strip() + '\t' + '\t'.join([str(round(i,4)) for i in l1])
        f1.write(line+'\n')
