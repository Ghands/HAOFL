# coding:utf-8

import sys
import argparse
import random
import numpy
import torch
import logging
import math
import os

import torch.nn as nn

from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel

from models import *
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, TrainDataset
from config import log_dir, with_position_models

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Runner(object):
    def __init__(self, opt, dtl_param):
        self.opt = opt

        if 'bert' in self.opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, 'bert-base-uncased')
            # TODO: Add the instance of bert based model
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='normal_tokenizer.dat'
            )
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_embedding_matrix.dat'.format(str(opt.embed_dim))
            )
            self.model = opt.model_class(opt, tokenizer, embedding_matrix).to(opt.device)

        self.train_set = TrainDataset(opt.dataset_file['train'], tokenizer, opt, opt.dtl_method, dtl_param,
                                      opt.name_tail)
        self.val_set = TrainDataset(opt.dataset_file['test'], tokenizer, opt, opt.dtl_method, dtl_param, opt.name_tail)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                if self.opt.model_name in with_position_models:
                    inputs = {"text": sample_batched["text"].to(self.opt.device),
                              'aspect': sample_batched['aspect'].to(self.opt.device),
                              'position': sample_batched['position'].to(self.opt.device)}
                else:
                    inputs = {"text": sample_batched["text"].to(self.opt.device),
                              "aspect": sample_batched['aspect'].to(self.opt.device)}
                outputs = self.model(self.opt.dtl_method, self.opt.dpl_mode, train=True, inputs=inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_val_acc{1}'.format(self.opt.model_name, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                if self.opt.model_name in with_position_models:
                    t_inputs = {"text": t_sample_batched["text"].to(self.opt.device),
                                'aspect': t_sample_batched['aspect'].to(self.opt.device),
                                'position': t_sample_batched['position'].to(self.opt.device)}
                else:
                    t_inputs = {"text": t_sample_batched["text"].to(self.opt.device),
                                "aspect": t_sample_batched['aspect'].to(self.opt.device)}
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(self.opt.dtl_method, self.opt.dpl_mode, train=True, inputs=t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return acc, f1

    def run(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.val_set, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1 = self._evaluate_acc_f1(val_data_loader)  # val dataset is the test dataset
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='baseline', type=str, help='The name of running model')

    # Parameters about DTL layer
    parser.add_argument('--dtl_method', default='filter', type=str,
                        help='The data transformation method used in DTL layer, possible values are: splitting, '
                             'sliding, filter')
    parser.add_argument('--split_size', default=400, type=int,
                        help='The size of text slices that obtained by splitting window')
    parser.add_argument('--slide_size', default=400, type=int,
                        help='The size of text slices that obtained by sliding window')
    parser.add_argument('--stride_size', default=200, type=int, help='The size of step when using sliding window')
    parser.add_argument('--sentence_num', default=3, type=int,
                        help='The max number of sentences of each text slice when using text filter')
    parser.add_argument('--text_slice_num', default=5, type=int,
                        help='The number of text slices that obtained by text filter')

    # Parameters about DPL layer
    parser.add_argument('--dpl_mode', default='encoder', type=str,
                        help='The mode that DPL layer takes to process text slices, available values '
                             'are: encoder, analysis')

    # Usual hyper-parameters
    parser.add_argument('--embed_dim', default=300, type=int, help='The dimension of embedding')
    parser.add_argument('--hidden_dim', default=300, type=int, help='The dimension of hidden state')
    parser.add_argument('--polarities_dim', default=3, type=int, help='The dimension of logits')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--log_step', default=5, type=int, help='The interval of printing results')
    parser.add_argument('--max_seq_len', default=200, type=int, help='The sequence length of each window')
    parser.add_argument('--device', default=None, type=str, help='The device to run model, e.g. cuda:1')
    parser.add_argument('--seed', default=None, type=int, help='The random seed')
    # parser.add_argument('--n_gpu', default=1, type=int, help='The number of gpus to support data parallel')
    # parser.add_argument('--max_bert_len', default=4000, type=int,
    #                     help='The max length of a single text to input when bert used')
    parser.add_argument('--hops', default=3, type=int, help='A hyper-parameter for MemNet and RAM')

    # Parameters about dataset
    parser.add_argument('--max_aspect_len', default=30, type=int, help='The max number of words that consist of aspect')
    parser.add_argument('--aspect_pos_len', default=40, type=int,
                        help='The position list of the aspect appears in text')
    parser.add_argument('--batch', default=True, type=bool, help='Running model with mini-batch or not')
    parser.add_argument('--fix_max_len', default=4000, type=int, help='The max words of full document')
    parser.add_argument('--name_tail', default='', type=str, help='Short comment defined by user')

    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'baseline': BaselineHAOFL,
        'atae': ATAEHAOFL,
        'ian': IANHAOFL,
        'memnet': MemNetHAOFL,
        'ram': RAMHAOFL,
        'tnet': TNETHAOFL,
        'aoa': AOAHAOFL,
        'mgan': MGANHAOFL
    }

    dataset_files = {
        "document-level": {
            "train": "./datasets/train.raw",
            "test": "./datasets/test.raw"
        }
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files['document-level']
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.dtl_method == "splitting":
        dtl_param = opt.split_size
    elif opt.dtl_method == "sliding":
        dtl_param = '{}@{}'.format(opt.slide_size, opt.stride_size)
    elif opt.dtl_method == 'filter':
        dtl_param = '{}@{}'.format(opt.sentence_num, opt.text_slice_num)
    else:
        raise ValueError("The value of `trans_method` is not supported!")
    log_file = '{}/{}_{}-{}_{}.log'.format(log_dir, opt.model_name, opt.dtl_method, dtl_param,
                                           strftime("%y%m%d-%H%M", localtime()))

    fmt = '%(asctime)s: %(message)s'
    format_str = logging.Formatter(fmt)
    log_fh = logging.FileHandler(log_file)
    log_fh.setFormatter(format_str)
    logger.addHandler(log_fh)

    runner = Runner(opt, dtl_param)
    runner.run()


if __name__ == '__main__':
    main()
