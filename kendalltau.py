# -*- coding:utf-8 _*-

import numpy as np
import pandas as pd

def compute_kenall(list_0, list_1):
    lens = len(list_0)
    conc, disc = 0, 0
    for i in range(lens):
        for j in range(i + 1, lens):
            if list_0[i] == list_0[j]:
                continue
            elif np.sign(list_0[i] - list_0[j]) * np.sign(list_1[i] - list_1[j])==1:
                conc += 1
            else:
                disc += 1

    return (conc - disc) / (conc + disc)


def compute_kendall(
    hyp1_scores: list, hyp2_scores: list, data: pd.DataFrame
) -> (int, list):
    """ Computes the official WMT19 shared task Kendall correlation score. """
    assert len(hyp1_scores) == len(hyp2_scores) == len(data)
    conc, disc = 0, 0
    for i, row in data.iterrows():
        if hyp1_scores[i] > hyp2_scores[i]:
            conc += 1
        else:
            disc += 1

    return (conc - disc) / (conc + disc)

