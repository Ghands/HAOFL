# coding: utf-8

import torch
import torch.nn as nn

from components import DynamicLSTM, Attention
from layers import DPLLayer
from frameworks import HAOFL


class IANDPL(DPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(IANDPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')

    def ian_kernel(self, x, aspect_x, x_len, aspect_len):
        x = self.embed(x)
        aspect_x = self.embed(aspect_x)
        x, (_, _) = self.lstm_context(x, x_len)
        aspect_x, (_, _) = self.lstm_aspect(aspect_x, aspect_len)

        aspect_len = aspect_len.float()
        aspect_xm = torch.div(torch.sum(aspect_x, dim=1, dtype=torch.float).to(self.opt.device),
                              aspect_len.view(aspect_len.size(0), 1))
        x_len = x_len.float()
        xm = torch.div(torch.sum(x, dim=1, dtype=torch.float).to(self.opt.device), x_len.view(x_len.size(0), 1))

        aspect_xa, _ = self.attention_aspect(aspect_x, xm)
        aspect_xa = aspect_xa.squeeze(dim=1)
        xa, _ = self.attention_aspect(x, aspect_xm)
        xa = xa.squeeze(dim=1)
        x = torch.cat((xa, aspect_xa), dim=-1)

        return x

    def encoder_mode(self, text_slices, aspect_tokens):
        x_len = torch.sum(text_slices != 0, dim=-1).to(self.opt.device)
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1).to(self.opt.device)

        return self.ian_kernel(text_slices, aspect_tokens, x_len, aspect_len)

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1).to(self.opt.device)

        return self.ian_kernel(text_slices, aspect_tokens, x_len, aspect_len)


class IANHAOFL(HAOFL):
    """A model constructed with HAOFL framework, IAN is the model used in DPL layer, and LSTM is used in SAL layer."""
    def __init__(self, opt, tokenizer, embedding_matrix):
        super(IANHAOFL, self).__init__(opt, tokenizer, embedding_matrix, 2)

    def set_dpl(self, embedding_matrix, opt):
        return IANDPL(embedding_matrix, opt)
