# coding:utf-8

import torch
import torch.nn as nn

from transformers import BertModel

from layers import DPLLayer
from frameworks import HAOFL


class BertDPL(DPLLayer):
    def __init__(self, opt):
        super(BertDPL, self).__init__(opt)

        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='./pretrained', force_download=False)
        self.cls_indices = torch.full((1, 1), 101, dtype=torch.int64)
        self.sep_indices = torch.full((1, 1), 102, dtype=torch.int64)

    def encoder_mode(self, text_slices, aspect_tokens):
        x_shape = text_slices.size()
        cls_indices = self.cls_indices.expand(x_shape[0], 1).to(self.opt.device)
        sep_indices = self.sep_indices.expand(x_shape[0], 1).to(self.opt.device)
        x = torch.cat((cls_indices, text_slices, sep_indices, aspect_tokens, sep_indices), dim=-1).to(self.opt.device)
        segments = torch.zeros(x.size(), dtype=torch.int64).to(self.opt.device)
        segments[:, x_shape[1] +2:] = 1
        _, x = self.bert(x, token_type_ids=segments)

        return x

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        now_x_shape = text_slices.size()
        now_aspect_shape = aspect_tokens.size()
        if now_x_shape[1] + now_aspect_shape[1] > 509:
            raise ValueError("The total size of text slices is too large for BERT to handle correctly, "
                             "please set smaller `sentence_num` or `text_slice_num`!")

        cls_indices = self.cls_indices.expand(x_shape[0], 1).to(self.opt.device)
        sep_indices = self.sep_indices.expand(x_shape[0], 1).to(self.opt.device)
        x = torch.cat((cls_indices, text_slices, sep_indices, aspect_tokens, sep_indices), dim=-1).to(self.opt.device)
        segments = torch.zeros(x.size(), dtype=torch.int64).to(self.opt.device)
        segments[:, now_x_shape[1] + 2:] = 1
        empty = torch.zeros(x.size(), dtype=torch.int64).to(self.opt.device)
        empty[x != 0] = 1
        _, x = self.bert(x, attention_mask=empty, token_type_ids=segments)

        return x


class BertHAOFL(HAOFL):
    """A model constructed with HAOFL framework, BERT is the model used in DPL layer, and LSTM is used in SAL layer."""
    def __init__(self, opt, tokenizer):
        super(BertHAOFL, self).__init__(opt, tokenizer, None, 1)

    def set_dpl(self, embedding_matrix, opt):
        return BertDPL(opt)
