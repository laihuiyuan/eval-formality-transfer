# -*- coding: utf-8 -*-

import os
import sys
import nltk
import gensim
import bert_score
from rouge import Rouge
from bleurt import score
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)
from comet.models import download_model
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

rouge = Rouge()
smooth = SmoothingFunction()

path = 'checkpoints/GoogleNews-vectors-negative300.bin.gz'
w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
BLEURT = score.BleurtScorer('checkpoints/bleurt-large-512')
BERT = bert_score.BERTScorer(lang='en', rescale_with_baseline=True)
COMET = download_model("checkpoints/wmt-large-da-estimator-1719")

data = []

with open(sys.argv[1], 'r') as f0, \
        open(sys.argv[2], 'r') as f1, \
        open(sys.argv[3], 'r') as f2:
    for l0, l1, l2 in zip(f0.readlines(), f1.readlines(), f2.readlines()):
        temp = []
        l0, l1, l2 = l0.strip(), l1.strip(), l2.strip()
        inputs = [{"src": l0, "mt": l1, "ref": l2}]
        temp.append(round(COMET.predict(inputs, cuda=True, show_progress=False)[-1][0], 4))
        inputs = [{"src": "", "mt": l1, "ref": l2}]
        temp.append(round(COMET.predict(inputs, cuda=True, show_progress=False)[-1][0], 4))
        temp.append(round(BLEURT.score(references=[l2], candidates=[l1])[0], 4))
        p, r, f = BERT.score([l1], [l2])
        temp.append(round(f[0].tolist(), 4))
        temp.append(round(nltk.meteor([l2], l1), 4))

        l1 = nltk.word_tokenize(l1)
        l2 = nltk.word_tokenize(l2)
        temp.append(round(w2v.wmdistance(l1, l2), 4))
        temp.append(round(sentence_bleu([l2], l1,
                                        smoothing_function=smooth.method1), 4))
        rouge_score = rouge.get_scores(' '.join(l1), ' '.join(l2), avg=True)
        temp.append(round(rouge_score['rouge-1']['f'], 4))
        temp.append(round(rouge_score['rouge-2']['f'], 4))
        temp.append(round(rouge_score['rouge-l']['f'], 4))
        data.append(temp)

with open(sys.argv[4], 'w') as f:
    for line in data:
        f.write('\t'.join([str(i) for i in line]) + '\n')
