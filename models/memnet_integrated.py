# coding:utf-8

import torch
import torch.nn as nn

from components import Attention, SqueezeEmbedding
from layers import NoAspectDTLLayer, DPLLayer
from frameworks import HAOFL


class MemNetDPL(DPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(MemNetDPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embed = SqueezeEmbedding()
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(opt.embed_dim, opt.hidden_dim)

    def memnet_kernel(self, x, aspect_x, x_len, aspect_len):
        x = self.embed(x)
        x = self.squeeze_embed(x, x_len)
        aspect_x = self.embed(aspect_x)
        aspect_x = torch.div(torch.sum(aspect_x, dim=1, dtype=torch.float).to(self.opt.device), aspect_len.view(aspect_len.size(0), 1))

        xs = aspect_x.unsqueeze(dim=1)
        for _ in range(self.opt.hops):
            xs = self.x_linear(xs)
            xsa, _ = self.attention(x, xs)
            xs = xsa + xs
        x = xs.view(x.size(0), -1)

        return x

    def encoder_mode(self, text_slices, aspect_tokens):
        x_len = torch.sum(text_slices != 0, dim=-1)
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1, dtype=torch.float).to(self.opt.device)

        return self.memnet_kernel(text_slices, aspect_tokens, x_len, aspect_len)

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1, dtype=torch.float).to(self.opt.device)

        return self.memnet_kernel(text_slices, aspect_tokens, x_len, aspect_len)


class MemNetHAOFL(HAOFL):
    def __init__(self, opt, tokenizer, embedding_matrix):
        super(MemNetHAOFL, self).__init__(opt, tokenizer, embedding_matrix, 1)

        self.dtl = NoAspectDTLLayer(opt, tokenizer)

    def set_dpl(self, embedding_matrix, opt):
        return MemNetDPL(embedding_matrix, opt)
