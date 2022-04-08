# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
from torch import cuda

from transformers import logging
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel


logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def cal_ppl(model, opt, seqs, f, loss_fn, tokenizer):
    ppl_all = []
    with torch.no_grad():
        for idx in range(0, len(seqs), opt.batch_size):
            inp = tokenizer.batch_encode_plus(
                        seqs[idx: idx+opt.batch_size],
                        padding=True, return_tensors='pt')
            src = inp['input_ids'].to(device)
            logits = model(src)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = src[..., 1:].contiguous()
            for x,y in zip(shift_logits, shift_labels):
                loss = loss_fn(x.view(-1, shift_logits.size(-1)),
                               y.view(-1))
                ppl_batch = torch.exp(loss).cpu().tolist()
                ppl_all.append(ppl_batch)
    for line, ppl in zip(seqs, ppl_all):
        line = line.strip() + '\t' + str(round(ppl, 6))
        f.write(line+'\n')


def main():
    parser = argparse.ArgumentParser('Fine-tuning GPT-2 for text style transfer')
    parser.add_argument('-order', default='BART', type=str, help='the model name')
    parser.add_argument('-model', default=0, type=str, help='the evaluated model name')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-dataset', default='xformal', type=str, help='the dataset name')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)
    special_tokens = [{'bos_token': '<bos>'}]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    for x in special_tokens:
        tokenizer.add_special_tokens(x)
    tokenizer.pad_token = tokenizer.eos_token

    model_0 = GPT2LMHeadModel.from_pretrained('gpt2')
    model_0.resize_token_embeddings(len(tokenizer))
    model_0.load_state_dict(torch.load('checkpoints/gpt2_for.chkpt'))
    model_0.to(device).eval()
    
    model_1 = GPT2LMHeadModel.from_pretrained('gpt2')
    model_1.resize_token_embeddings(len(tokenizer))
    model_1.load_state_dict(torch.load('checkpoints/gpt2_inf.chkpt'))
    model_1.to(device).eval()
    
    loss_fn =nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)

    seqs_0, seqs_1 = [], []
    with open('data/outputs/{}.human.txt'.format(opt.model), 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')[2]
            if i < 40:
                seqs_0.append(line.strip())
            else:
                seqs_1.append(line.strip())
    print('[Info] {} instances in total.'.format(len(seqs_0)))
    print('[Info] {} instances in total.'.format(len(seqs_1)))
    f = open('data/outputs/{}.ppl.txt'.format(
        opt.model, opt.dataset), 'w')
    cal_ppl(model_0, opt, seqs_0, f, loss_fn, tokenizer)
    cal_ppl(model_1, opt, seqs_1, f, loss_fn, tokenizer)


if __name__ == "__main__":
    main()
