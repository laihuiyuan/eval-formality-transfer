# -*- coding: utf-8 -*-

import os
import argparse

import torch
from torch import cuda
import torch.nn.functional as F
from transformers import logging
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('Evaluating the style strength of sentences')
    parser.add_argument('-model', default=0, type=str, help='the evaluated model name')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-batch_size', default=128, type=int, help='max sents in a batch')
    parser.add_argument('-dataset', default='xformal', type=str, help='the dataset name')
    parser.add_argument('-task', default='cls', type=str, help='classifier or regressior')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    if opt.prob == 'regression':
        num_label = 1
    else:
        num_label = 2

    config = BertConfig.from_pretrained('bert-base-cased',
                                        problem_type=opt.prob,
                                        num_labels=num_label)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                                                          config=config)
    model_dir = 'checkpoints/bert_{}_{}.chkpt'.format(opt.dataset, opt.prob[:2])
    model.load_state_dict(torch.load(model_dir))
    model.to(device).eval()

    f0 = open('data/outputs/{}.txt'.format(opt.model), 'r')
    f1 = open('data/outputs/{}.bert.{}.{}.txt'.format(
        opt.model, opt.dataset, opt.prob[:2]), 'w')
    with torch.no_grad():
        for i, line in enumerate(f0.readlines()):
            sentence = line.strip()
            inp = tokenizer.batch_encode_plus(
                [sentence],
                padding=True,
                return_tensors='pt')
            src = inp['input_ids'].to(device)
            mask = inp['attention_mask'].to(device)
            outs = model(src, mask)
            logits = outs.logits[0]
            if opt.prob == 'regression':
                line = line.strip() + '\t' + str(round(logits[0].item(), 4))
            else:
                logits = F.softmax(logits, dim=-1)
                line = line.strip() + '\t' + str(round(logits[1].item(), 4))
            f1.write(line + '\n')
    f0.close()
    f1.close()


if __name__ == '__main__':
    main()
