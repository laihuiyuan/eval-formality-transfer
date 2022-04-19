# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

import torch
from torch import cuda
from transformers import logging
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from utils.optim import ScheduledOptim

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def read_insts(dataset, prefix, tokenizer):
    src = 'data/{}/{}.0'.format(dataset, prefix, )
    tgt = 'data/{}/{}.1'.format(dataset, prefix, )

    seqs, label = [], []
    for i, x in enumerate([src, tgt]):
        with open(x, 'r') as f:
            for line in f.readlines():
                seq = tokenizer.encode(line[:130].strip())
                seqs.append(seq)
                label.append(i)

    return seqs, label


def read_csv(dataset, prefix, tokenizer):
    inp_file = 'data/{}/{}.csv'.format(dataset, prefix)

    data = pd.read_csv(inp_file).values

    label = data[:, 1].tolist()
    seqs = []
    for line in data[:, -1]:
        seqs.append(tokenizer.encode(line[:130].strip()))

    return seqs, label


def collate_fn(insts, pad_token_id=50256):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq

class SCDataset(torch.utils.data.Dataset):
    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def SCIterator(seqs, label, pad_id, shuffle, opt):
    '''Data iterator for style classifier'''

    def cls_fn(insts):
        src, tgt = list(zip(*insts))
        src = collate_fn(src, pad_id)
        if opt.prob == 'regression':
            tgt = torch.FloatTensor(tgt)
        else:
            tgt = torch.LongTensor(tgt)

        return (src, tgt)

    loader = torch.utils.data.DataLoader(
        SCDataset(
            insts=seqs,
            label=label),
        num_workers=2,
        shuffle=shuffle,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


def evaluate(model, valid_loader, epoch, tokenizer):
    '''Evaluation function for style classifier'''
    model.eval()
    corre_num = 0.
    total_num = 0.
    loss_list = []
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())
    model.train()
    print('[Info] Epoch {:02d}-valid: {}'.format(
        epoch, 'acc {:.4f} | loss {:.4f}').format(
        corre_num / total_num, np.mean(loss_list)))

    return corre_num / total_num, np.mean(loss_list)


def main():
    parser = argparse.ArgumentParser('The style classifier/regressor based on BERT')
    parser.add_argument('-lr', default=1e-5, type=float, help='the initial learning rate')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=100, type=int, help='print log every x steps')
    parser.add_argument('-dataset', default='xformal', type=str, help='the name of dataset')
    parser.add_argument('-eval_step', default=1000, type=int, help='early stopping training')
    parser.add_argument('-batch_size', default=32, type=int, help='maximum sents in a batch')
    parser.add_argument('-epoch', default=80, type=int, help='force stop at specified epoch')
    parser.add_argument('-task',
                        default='single_label_classification',
                        type=str, help='or regression')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)
    print('[Info]', opt)

    if opt.task == 'regression':
        num_label = 1
    else:
        num_label = 2
    config = BertConfig.from_pretrained(
        'bert-base-cased',
        problem_type=opt.task,
        num_labels=num_label)

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-cased')
    pad_id = tokenizer.pad_token_id

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        config=config)
    model.to(device).train()

    print('[Info] Built a model with {} parameters'.format(
        sum(p.numel() for p in model.parameters())))

    if opt.prob == 'regression':
        train_src, train_tgt = read_csv(opt.dataset, 'train', tokenizer)
        valid_src, valid_tgt = read_csv(opt.dataset, 'valid', tokenizer)
    else:
        train_src, train_tgt = read_insts(opt.dataset, 'train', tokenizer)
        valid_src, valid_tgt = read_insts(opt.dataset, 'valid', tokenizer)
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))
    train_loader = SCIterator(train_src, train_tgt, pad_id, True, opt)
    valid_loader = SCIterator(valid_src, valid_tgt, pad_id, False, opt)

    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr, 500)

    tab = 0
    eval_loss = 1e9
    corre_num = 0.
    total_num = 0.
    loss_list = []
    start = time.time()
    for epoch in range(opt.epoch):

        for batch in train_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            y_hat = logits.argmax(dim=-1)
            same = [int(p == q) for p, q in zip(tgt, y_hat)]
            corre_num += sum(same)
            total_num += len(tgt)
            loss_list.append(loss.item())

            if optimizer.cur_step % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] Epoch {:02d}-{:05d}: | acc {:.4f} | '
                      'loss {:.4f} | lr {:.6f} | second {:.2f}'.format(
                    epoch, optimizer.cur_step, corre_num / total_num,
                    np.mean(loss_list), lr, time.time() - start))
                corre_num = 0.
                total_num = 0.
                loss_list = []
                start = time.time()

            if ((len(train_loader) >= opt.eval_step
                 and optimizer.cur_step % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step
                        and optimizer.cur_step % len(train_loader) == 0)):
                valid_acc, valid_loss = evaluate(model, valid_loader, epoch, tokenizer)
                if eval_loss > valid_loss:
                    eval_loss = valid_loss
                    save_path = 'checkpoints/bert_{}_{}.chkpt'.format(
                        opt.prob[:3], opt.dataset)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    tab = 0
                else:
                    tab += 1
                    if tab == 5:
                        exit()

if __name__ == '__main__':
    main()
