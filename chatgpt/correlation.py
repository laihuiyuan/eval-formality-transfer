# -*- coding:utf-8 _*-

import math
import json
import numpy as np
from tabulate import tabulate
from scipy.stats import pearsonr, spearmanr, kendalltau


def pair_acc(input_file, metrics, to_formal, use_ref):
    metric_and_corr = []
    headers = ['metric', 'pairwise acc']

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

    for metric in metrics:

        aut_scores = []
        hum_scores = []
        for i, line in enumerate(data):

            aut_temp = []
            hum_temp = []
            if (to_formal and i < 40) or (not to_formal and i >= 40) or to_formal == 'all':

                for model in line['sys']:
                    if not use_ref and model == 'ref':
                        continue
                    if metric == 'wmd' or metric == 'gpt2_ppl':
                        aut_temp.append(-line['sys'][model][metric])
                    else:
                        aut_temp.append(line['sys'][model][metric])
                    # IAA score
                    # hum_temp.append(line['sys'][model]['human_1'])
                    hum_temp.append((line['sys'][model]['human_1'] + line['sys'][model]['human_2']) / 2)
                aut_scores.append(aut_temp)
                hum_scores.append(hum_temp)


        aut_scores = np.array(aut_scores).mean(0)
        hum_scores = np.array(hum_scores).mean(0)
        # print(aut_scores)
        # print(hum_scores)
        num = 0
        all = 0
        for i in range(len(aut_scores) - 1):
            for j in range(i + 1, len(aut_scores)):
                all += 1
                # print(np.sign(aut_scores[i]-aut_scores[j]), np.sign(hum_scores[i]-hum_scores[j]))
                if np.sign(aut_scores[i] - aut_scores[j]) == np.sign(hum_scores[i] - hum_scores[j]):
                    num += 1
                # else:
                #     print(i,j)
        # print(num, all)
        metric_and_corr.append([metric] + [num / all])

    print(tabulate(metric_and_corr, headers=headers, tablefmt='simple'))


def corr_system(input_file, metrics, to_formal, use_ref):
    metric_and_corr = []
    headers = ['metric', 'pearsonr', 'spearman', 'kendalltau']

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

    for metric in metrics:

        aut_scores = []
        hum_scores = []
        for i, line in enumerate(data):

            aut_temp = []
            hum_temp = []
            if (to_formal and i < 40) or (not to_formal and i >= 40) or to_formal == 'all':

                for model in line['sys']:
                    if not use_ref and model == 'ref':
                        continue
                    if metric == 'wmd' or metric == 'gpt2_ppl':
                        aut_temp.append(-line['sys'][model][metric])
                    else:
                        aut_temp.append(line['sys'][model][metric])
                    # IAA score
                    # hum_temp.append(line['sys'][model]['human_1'])
                    hum_temp.append((line['sys'][model]['human_1'] + line['sys'][model]['human_2']) / 2)
                aut_scores.append(aut_temp)
                hum_scores.append(hum_temp)

        aut_scores = np.array(aut_scores)
        hum_scores = np.array(hum_scores)
        corrs = [
            pearsonr(aut_scores.mean(0), hum_scores.mean(0))[0],
            spearmanr(aut_scores.mean(0), hum_scores.mean(0))[0],
            kendalltau(aut_scores.mean(0), hum_scores.mean(0))[0],
        ]
        metric_and_corr.append([metric] + corrs)

    print(tabulate(metric_and_corr, headers=headers, tablefmt='simple'))


def corr_sample(input_file, metrics, to_formal, use_ref):
    metric_and_corr = []
    headers = ['metric', 'pearsonr', 'spearman', 'kendalltau']

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

    for metric in metrics:
        corrs = []

        for i, line in enumerate(data):

            if (to_formal and i < 40) or (not to_formal and i >= 40) or to_formal == 'all':
                aut_scores = []
                hum_scores = []
                for model in line['sys']:
                    if not use_ref and model == 'ref':
                        continue
                    if metric == 'wmd' or metric == 'gpt2_ppl':
                        aut_scores.append(-line['sys'][model][metric])
                    else:
                        aut_scores.append(line['sys'][model][metric])
                    # IAA score
                    # hum_scores.append(line['sys'][model]['human_1'])
                    hum_scores.append((line['sys'][model]['human_1'] + line['sys'][model]['human_2']) / 2)

                temp = [
                    pearsonr(aut_scores, hum_scores)[0],
                    spearmanr(aut_scores, hum_scores)[0],
                    kendalltau(aut_scores, hum_scores)[0],
                ]
                temp = [x if not math.isnan(x) else 0 for x in temp]
                corrs.append(temp)
        corrs = np.array(corrs)
        pear, sper, ktau = np.mean(corrs[:, 0]), np.mean(corrs[:, 1]), np.mean(corrs[:, 2])
        metric_and_corr.append([metric, pear, sper, ktau])

    print(tabulate(metric_and_corr, headers=headers, tablefmt='simple'))


def corr_dataset(input_file, metrics, to_formal, use_ref):
    metric_and_corr = []
    headers = ['metric', 'pearsonr', 'spearman', 'kendalltau']

    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

    for metric in metrics:

        aut_scores = []
        hum_scores = []

        for i, line in enumerate(data):

            if (to_formal and i < 40) or (not to_formal and i >= 40) or to_formal == 'all':
                for model in line['sys']:
                    if not use_ref and model == 'ref':
                        continue
                    if metric == 'wmd' or metric == 'gpt2_ppl':
                        aut_scores.append(-line['sys'][model][metric])
                    else:
                        aut_scores.append(line['sys'][model][metric])
                    # IAA score
                    # hum_scores.append(line['sys'][model]['human_1'])
                    hum_scores.append((line['sys'][model]['human_1'] + line['sys'][model]['human_2']) / 2)

        corrs = ([
            pearsonr(aut_scores, hum_scores)[0],
            spearmanr(aut_scores, hum_scores)[0],
            kendalltau(aut_scores, hum_scores)[0],
        ])
        metric_and_corr.append([metric] + corrs)

    print(tabulate(metric_and_corr, headers=headers, tablefmt='simple'))


# metric = ['human_2']

use_ref = True
metric = ['bleu', 'rouge-1', 'rouge-2', 'rouge-l', 'chrf',
          'wmd', 'meteor', 'bertscore', 'bleurt', 'comet-20',
          'comet-22', 'chatgpt', 'chatgpt_multi']
pair_acc('content.src.json', metric, True, use_ref)
pair_acc('content.src.json', metric, False, use_ref)
# pair_acc('content.src.json', metric, 'all', use_ref)

# corr_system('content.src.json', metric , True, use_ref)
# corr_system('content.src.json', metric, False, use_ref)
# corr_system('content.src.json', metric, 'all', use_ref)

corr_sample('content.src.json', metric, True, use_ref)
corr_sample('content.src.json', metric, False, use_ref)
corr_sample('content.src.json', metric, 'all', use_ref)

corr_dataset('content.src.json', metric, True, use_ref)
corr_dataset('content.src.json', metric, False, use_ref)
corr_dataset('content.src.json', metric, 'all', use_ref)

metric = ['r-pt16', 'c-pt16', 'c-gyafc', 'chatgpt', 'chatgpt_multi']
pair_acc('style.json', metric, True, use_ref)
pair_acc('style.json', metric, False, use_ref)
# pair_acc('style.json', metric, 'all', use_ref)

# # corr_system('style.json', metric , True, use_ref)
# # corr_system('style.json', metric, False, use_ref)
# # corr_system('style.json', metric, 'all', use_ref)

corr_sample('style.json', metric, True, use_ref)
corr_sample('style.json', metric, False, use_ref)
corr_sample('style.json', metric, 'all', use_ref)

corr_dataset('style.json', metric, True, use_ref)
corr_dataset('style.json', metric, False, use_ref)
corr_dataset('style.json', metric, 'all', use_ref)

metric = ['gpt2_ppl', 'chatgpt', 'chatgpt_multi']
pair_acc('fluency.json', metric, True, use_ref)
pair_acc('fluency.json', metric, False, use_ref)
# pair_acc('fluency.json', metric, 'all', use_ref)

# corr_system('fluency.json', metric, True, use_ref)
# corr_system('fluency.json', metric, False, use_ref)
# corr_system('fluency.json', metric, 'all', use_ref)

corr_sample('fluency.json', metric, True, use_ref)
corr_sample('fluency.json', metric, False, use_ref)
corr_sample('fluency.json', metric, 'all', use_ref)

corr_dataset('fluency.json', metric, True, use_ref)
corr_dataset('fluency.json', metric, False, use_ref)
corr_dataset('fluency.json', metric, 'all', use_ref)
