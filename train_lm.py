# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from torch.nn.utils import clip_grad_norm_

from transformers import logging
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from utils.optim import ScheduledOptim

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def read_insts(dataset, style, prefix, tokenizer):
    file = 'data/{}/{}.{}'.format(dataset, prefix,style)

    seqs = []
    start = time.time()
    with open(file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            seq_id = tokenizer.encode(line.strip()[:80])
            seqs.append(seq_id)
    del tokenizer

    return seqs

def collate_fn(insts, pad_token_id=50256):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


def paired_collate_fn(insts):
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)

    return src_inst, tgt_inst


class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, src_inst=None, tgt_inst=None):
        self._src_inst = src_inst
        self._tgt_inst = tgt_inst

    def __len__(self):
        return len(self._src_inst)

    def __getitem__(self, idx):
        return self._src_inst[idx], self._tgt_inst[idx]


def GPT2Iterator(train_src, train_tgt, valid_src, valid_tgt, opt):
    '''Data iterator for fine-tuning GPT-2'''

    train_loader = torch.utils.data.DataLoader(
        GPT2Dataset(
            src_inst=train_src,
            tgt_inst=train_tgt),
        num_workers=12,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        GPT2Dataset(
            src_inst=valid_src,
            tgt_inst=valid_tgt),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader


def evaluate(model, loss_fn, valid_loader, tokenizer, epoch, step):
    '''Evaluation function for GPT-2'''
    model.eval()
    loss_list = []
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)
            logits = model(src)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
            loss_list.append(loss.item())

    model.train()
    print('[Info] {:02d}-{:06d} | loss {:.4f}'.format(
          epoch, step, np.mean(loss_list)))

    return np.mean(loss_list)


def main():
    parser = argparse.ArgumentParser('Fine-tuning GPT-2 for evaluating ppl')
    parser.add_argument('-lr', default=3e-5, type=float, help='initial earning rate')
    parser.add_argument('-epoch', default=30, type=int, help='force stop at 20 epoch')
    parser.add_argument('-acc_steps', default=1, type=int, help='accumulation_steps')
    parser.add_argument('-style', default=0, type=int, help='informal(0) vs formal(0)')
    parser.add_argument('-batch_size', default=128, type=int, help='the size in a batch')
    parser.add_argument('-dataset', default='xformal', type=str, help='the dataset name')
    parser.add_argument('-patience', default=3, type=int, help='early stopping fine-tune')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=100, type=int, help='print logs every x step')
    parser.add_argument('-eval_step', default=1000, type=int, help='evaluate every x step')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)
    special_tokens = [{'bos_token': '<bos>'}]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    for x in special_tokens:
        tokenizer.add_special_tokens(x)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device).train()

    train_src = read_insts(opt.dataset, opt.style, 'train', tokenizer)
    valid_src = read_insts(opt.dataset, opt.style, 'valid', tokenizer)
    train_tgt = train_src.copy()
    valid_tgt = valid_src.copy()

    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))

    train_loader, valid_loader = GPT2Iterator(train_src, train_tgt,
                                              valid_src, valid_tgt, opt)

    loss_fn =nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09), opt.lr, 500)

    step = 0
    loss_list = []
    start = time.time()
    tab, eval_loss = 0, 1e8
    for epoch in range(opt.epoch):
        for batch in train_loader:
            step += 1
            src, tgt = map(lambda x: x.to(device), batch)

            logits = model(src)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
            loss_list.append(loss.item())

            loss = loss/opt.acc_steps
            loss.backward()

            if step % opt.acc_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

            if step % opt.log_step == 0:
                lr = optimizer._optimizer.param_groups[0]['lr']
                print('[Info] {:02d}-{:06d} | loss {:.4f} | '
                      'lr {:.6f} | second {:.1f}'.format(epoch, step,
                      np.mean(loss_list), lr, time.time() - start))
                loss_list = []
                start = time.time()

            if ((len(train_loader) > opt.eval_step
                 and step % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step
                        and step % len(train_loader) == 0)):
                valid_loss = evaluate(model, loss_fn, valid_loader, tokenizer, epoch, step)
                if eval_loss >= valid_loss:
                    save_path = 'checkpoints/gpt2_{}_{}.chkpt'.format(
                        opt.dataset, opt.style)
                    torch.save(model.state_dict(), save_path)
                    print('[Info] The checkpoint file has been updated.')
                    eval_loss = valid_loss
                    tab = 0
                else:
                    tab += 1
                if tab == opt.patience:
                    exit()


if __name__ == "__main__":
    main()
