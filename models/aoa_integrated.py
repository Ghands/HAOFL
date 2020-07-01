# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from components import DynamicLSTM
from layers import DPLLayer
from frameworks import HAOFL


class AOADPL(DPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(AOADPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def aoa_kernel(self, x, aspect_x, x_len, aspect_len):
        x = self.embed(x)
        aspect_x = self.embed(aspect_x)
        x, (_, _) = self.ctx_lstm(x, x_len)
        aspect_x, (_, _) = self.asp_lstm(aspect_x, aspect_len)

        new_x = torch.matmul(x, torch.transpose(aspect_x, 1, 2))
        alpha = F.softmax(new_x, dim=1)
        beta = F.softmax(new_x, dim=2).mean(dim=1, keepdim=True)
        gamma = torch.matmul(alpha, beta.transpose(1, 2))
        x = torch.matmul(torch.transpose(x, 1, 2), gamma).squeeze(-1)

        return x

    def encoder_mode(self, text_slices, aspect_tokens):
        x_len = torch.sum(text_slices != 0, dim=1)
        aspect_len = torch.sum(aspect_tokens != 0, dim=1)

        return self.aoa_kernel(text_slices, aspect_tokens, x_len, aspect_len)

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        aspect_len = torch.sum(aspect_tokens != 0, dim=1)

        return self.aoa_kernel(text_slices, aspect_tokens, x_len, aspect_len)


class AOAHAOFL(HAOFL):
    def __init__(self, opt, tokenizer, embedding_matrix):
        super(AOAHAOFL, self).__init__(opt, tokenizer, embedding_matrix, 2)

    def set_dpl(self, embedding_matrix, opt):
        return AOADPL(embedding_matrix, opt)
