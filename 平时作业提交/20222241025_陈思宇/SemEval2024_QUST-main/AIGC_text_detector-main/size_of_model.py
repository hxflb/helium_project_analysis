import os
import subprocess
import sys
from itertools import count
import multiprocessing
import numpy as np
from tqdm import tqdm
import argparse, random
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F # self added
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import *
from dataset import chatgpt_load_datasets
from utils import summary, distributed
from pu_loss_mod import pu_loss_auto as pu_loss

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(max_epochs=None,
        device=None,
        batch_size=24,
        max_sequence_length=128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        data_dir='data',
        real_dataset='webtext',
        fake_dataset='xl-1542M-nucleus',
        token_dropout=None,
        large=False,
        learning_rate=2e-5,
        weight_decay=0,
        **kwargs):
    # model_name = 'roberta-large' if large else 'roberta-base'
    model_name = kwargs['model_name']
    model_path = os.path.join(kwargs['local_model'], model_name) if kwargs[
                                                                    'local_model'] is not None else model_name  # self added: direct to pretrained model_dir

    tokenization_utils.logger.setLevel('ERROR')
    if model_name in ['distilbert-base-cased', 'distilbert-base-uncased']:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_name in ['chinese-roberta-wwm-ext', 'bert-base-cased',
                        'bert-base-uncased']:  # load chinese roberta with BERT
        tokenizer = BertTokenizer.from_pretrained(model_path)
        # model_config = BertConfig.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_name in ['roberta-base', 'roberta-large']:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    elif model_name in ['xlnet-base-cased']:
        tokenizer = XLNetTokenizer.from_pretrained(model_path)
        model = XLNetForSequenceClassification.from_pretrained(model_path).to(device)

    # 计算虚拟模型的参数数量
    num_params = count_parameters(model)
    print(f"模型中的参数数量为: {num_params}")

if __name__ == '__main__':
        from option import get_parser

        args, unparsed = get_parser()
        args.sentence_lengths = list()
        trained_on_what = ''
        trained_on_what_ls = args.train_data_file.split('/')  # record dataset to train on
        for i in range(len(trained_on_what_ls)):  # find a valid dataset name
            if '.' not in trained_on_what_ls[i]:
                trained_on_what = trained_on_what_ls[i]
                break

        # dir processing
        args.train_data_file = os.path.join(args.local_data, args.train_data_file)
        args.val_data_file = os.path.join(args.local_data, args.val_data_file)
        if args.val_file1 is not None:
            args.val_file1 = os.path.join(args.local_data, args.val_file1)
        if args.val_file2 is not None:
            args.val_file2 = os.path.join(args.local_data, args.val_file2)
        if args.val_file3 is not None:
            args.val_file3 = os.path.join(args.local_data, args.val_file3)
        if args.val_file4 is not None:
            args.val_file4 = os.path.join(args.local_data, args.val_file4)
        if args.val_file5 is not None:
            args.val_file5 = os.path.join(args.local_data, args.val_file5)
        if args.val_file6 is not None:
            args.val_file6 = os.path.join(args.local_data, args.val_file6)

        # automatically set save dir to args.log_dir
        if args.log_dir is None:  # train_summary/train_config/Aug+PUconfig
            fast_flag = 'FAST' if args.fast else ''
            clean_flag = 'CLEAN' if args.clean > 0 else ''
            if type(args.aug_mode) == list:
                aug_mode = '__'.join(args.aug_mode)
            else:
                aug_mode = args.aug_mode
            args.log_dir = f'./results/{args.model_name}{fast_flag}_{trained_on_what}{clean_flag}_{args.data_name}_{args.mode}_{args.seed}/{args.max_epochs if args.training_proportion is None else args.training_proportion}_{args.batch_size}_{args.learning_rate}_{args.weight_decay}/{aug_mode}_{args.aug_min_length}_{args.pu_type}_{args.lamb}_{args.prior}_{args.len_thres}'
        # if args.epoch_size is None:
        #     args.epoch_size = args.max_epochs

        print(f'ARGS: {args}')
        test(**dict(**vars(args), args=args))