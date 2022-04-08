# -*- coding: utf-8 -*-

import os
import sys
import nltk
import gensim
import sacrebleu
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

'''
NOTE: 
you have to download GoogleNews-vectors-negative300.bin.gz
and bleurt-large-512, and put them in checkpoints
'''

path = 'checkpoints/GoogleNews-vectors-negative300.bin.gz'
w2v = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
BLEURT = score.BleurtScorer('checkpoints/bleurt-large-512')
BERT = bert_score.BERTScorer(lang='en', rescale_with_baseline=True)
COMET = download_model("wmt-large-da-estimator-1719")

data = []
with open(sys.argv[1], 'r') as fin:
    for line in fin.readlines():
        sample = []
        line = line.strip().split('\t')
        sample.append(line)
        src, out, ref = line[1], line[2], line[3]

        # COMET score
        inputs = [{"src": src, "mt": out, "ref": ref}]
        sample.append(round(COMET.predict(inputs, cuda=True, show_progress=False)[-1][0], 4))

        # COMET without input settings
        inputs = [{"src": "", "mt": out, "ref": src}]
        sample.append(round(COMET.predict(inputs, cuda=True, show_progress=False)[-1][0], 4))
        inputs = [{"src": "", "mt": out, "ref": ref}]
        sample.append(round(COMET.predict(inputs, cuda=True, show_progress=False)[-1][0], 4))

        # BLEURT score
        sample.append(round(BLEURT.score(references=[src], candidates=[out])[0], 4))
        sample.append(round(BLEURT.score(references=[ref], candidates=[out])[0], 4))

        # BERT score
        _, _, f = BERT.score([out], [src])
        sample.append(round(f[0].tolist(), 4))
        _, _, f = BERT.score([out], [ref])
        sample.append(round(f[0].tolist(), 4))

        # METEOR score
        sample.append(round(nltk.meteor([src], out), 4))
        sample.append(round(nltk.meteor([ref], out), 4))

        # chrF score
        sample.append(sacrebleu.sentence_chrf(out, [src]))
        sample.append(sacrebleu.sentence_chrf(out, [ref]))

        # tokenization
        src = nltk.word_tokenize(src)
        out = nltk.word_tokenize(out)
        ref = nltk.word_tokenize(ref)

        # WMD score
        sample.append(round(w2v.wmdistance(out, src), 4))
        sample.append(round(w2v.wmdistance(out, ref), 4))

        # BLEU score
        sample.append(round(sentence_bleu([src], out,
                                          smoothing_function=smooth.method1), 4))
        sample.append(round(sentence_bleu([ref], out,
                                          smoothing_function=smooth.method1), 4))

        # ROUGE score
        rouge_score = rouge.get_scores(' '.join(out), ' '.join(src), avg=True)
        sample.append(round(rouge_score['rouge-1']['f'], 4))
        sample.append(round(rouge_score['rouge-2']['f'], 4))
        sample.append(round(rouge_score['rouge-l']['f'], 4))

        rouge_score = rouge.get_scores(' '.join(out), ' '.join(ref), avg=True)
        sample.append(round(rouge_score['rouge-1']['f'], 4))
        sample.append(round(rouge_score['rouge-2']['f'], 4))
        sample.append(round(rouge_score['rouge-l']['f'], 4))

        data.append(sample)

with open(sys.argv[4], 'w') as f:
    for line in data:
        f.write('\t'.join([str(i) for i in line]) + '\n')
