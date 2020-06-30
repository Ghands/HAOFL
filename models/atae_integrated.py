# coding:utf-8

import torch
import torch.nn as nn

from components import DynamicLSTM, SqueezeEmbedding, NoQueryAttention
from layers import DPLLayer
from frameworks import HAOFL


class ATAEDPL(DPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(ATAEDPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embed = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim + opt.embed_dim, score_function='bi_linear')

    def atae_kernel(self, x, aspect_x, x_len, x_len_max, aspect_len):
        x = self.embed(x)
        x = self.squeeze_embed(x, x_len)
        aspect_x = self.embed(aspect_x)
        aspect_x = torch.div(torch.sum(aspect_x, dim=1, dtype=torch.float).to(self.opt.device), aspect_len.view(aspect_len.size(0), 1))
        aspect_x = torch.unsqueeze(aspect_x, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((x, aspect_x), dim=-1)

        x, (_, _) = self.lstm(x, x_len)
        xa = torch.cat((x, aspect_x), dim=-1)
        _, score = self.attention(xa)
        del xa
        x = torch.squeeze(torch.bmm(score, x), dim=1)

        return x

    def encoder_mode(self, text_slices, aspect_tokens):
        x_len = torch.sum(text_slices != 0, dim=-1)
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1, dtype=torch.float).to(self.opt.device)
        x_len_max = torch.max(x_len)

        return self.atae_kernel(text_slices, aspect_tokens, x_len, x_len_max, aspect_len)

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1).to(self.opt.device)
        x_len_max = torch.max(x_len)

        return self.atae_kernel(text_slices, aspect_tokens, x_len, x_len_max, aspect_len)


class ATAEHAOFL(HAOFL):
    def __init__(self, opt, tokenizer, embedding_matrix):
        super(ATAEHAOFL, self).__init__(opt, tokenizer, embedding_matrix, 1)

    def set_dpl(self, embedding_matrix, opt):
        return ATAEDPL(embedding_matrix, opt)
