# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from components import DynamicLSTM
from layers import PositionDPLLayer
from frameworks import PositionHAOFL


class AlignmentMatrix(nn.Module):
    def __init__(self, opt):
        super(AlignmentMatrix, self).__init__()
        self.opt = opt
        self.w_u = nn.Parameter(torch.Tensor(6*opt.hidden_dim, 1))

    def forward(self, batch_size, ctx, asp):
        ctx_len = ctx.size(1)
        asp_len = asp.size(1)
        alignment_mat = torch.zeros(batch_size, ctx_len, asp_len).to(self.opt.device)
        ctx_chunks = ctx.chunk(ctx_len, dim=1)
        asp_chunks = asp.chunk(asp_len, dim=1)
        for i, ctx_chunk in enumerate(ctx_chunks):
            for j, asp_chunk in enumerate(asp_chunks):
                feat = torch.cat([ctx_chunk, asp_chunk, ctx_chunk*asp_chunk], dim=2) # batch_size x 1 x 6*hidden_dim
                alignment_mat[:, i, j] = feat.matmul(self.w_u.expand(batch_size, -1, -1)).squeeze(-1).squeeze(-1)
        return alignment_mat


class MGANDPL(PositionDPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(MGANDPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.w_a2c = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        self.w_c2a = nn.Parameter(torch.Tensor(2 * opt.hidden_dim, 2 * opt.hidden_dim))
        self.alignment = AlignmentMatrix(opt)

    def location_memory(self, x, x_len, all_pos_tuple):
        batch_size = x.size(0)
        seq_len = x.size(1)
        all_pos_tuple = all_pos_tuple.cpu().numpy()
        x_len = x_len.cpu().numpy()
        weight = list()

        for i in range(batch_size):
            weight_value = list()
            pos_tuple = all_pos_tuple[i]
            for j in range(0, len(pos_tuple), 2):
                if pos_tuple[j + 1] > seq_len:
                    continue
                if pos_tuple[j] != -1:
                    aspect_len = pos_tuple[j + 1] - pos_tuple[j]
                    weight_value.append(np.concatenate((1 - np.arange(pos_tuple[j], 0, -1) / (seq_len - aspect_len),
                                                        np.zeros(aspect_len),
                                                        1 - np.arange(1, seq_len - pos_tuple[j + 1] + 1) / (
                                                                seq_len - aspect_len))))
                else:
                    weight_value.append(np.ones(seq_len))
            weight_value = np.stack(weight_value).max(axis=0)

            weight.append(weight_value)

        x = torch.tensor(weight, dtype=torch.float).to(self.opt.device).unsqueeze(2) * x

        return x

    def mgan_kernel(self, x, aspect_x, x_len, aspect_len, pos_tuple, batch_size):
        x = self.embed(x)
        aspect_x = self.embed(aspect_x)

        x, (_, _) = self.ctx_lstm(x, x_len)
        x = self.location_memory(x, x_len, pos_tuple)
        x_pool = torch.sum(x, dim=1, dtype=torch.float).to(self.opt.device)
        x_pool = torch.div(x_pool, x_len.float().unsqueeze(-1)).unsqueeze(-1)

        aspect_x, (_, _) = self.asp_lstm(aspect_x, aspect_len)
        aspect_pool = torch.sum(aspect_x, dim=1, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.div(aspect_pool, aspect_len.float().unsqueeze(-1)).unsqueeze(-1)

        alignment_mat = self.alignment(batch_size, x, aspect_x)
        f_asp2ctx = torch.matmul(x.transpose(1, 2), F.softmax(alignment_mat.max(2, keepdim=True)[0], dim=1)).squeeze(-1)
        f_ctx2asp = torch.matmul(F.softmax(alignment_mat.max(1, keepdim=True)[0], dim=2), aspect_x).transpose(1,
                                                                                                              2).squeeze(
            -1)

        c_asp2ctx_alpha = F.softmax(x.matmul(self.w_a2c.expand(batch_size, -1, -1)).matmul(aspect_pool), dim=1)
        c_asp2ctx = torch.matmul(x.transpose(1, 2), c_asp2ctx_alpha).squeeze(-1)
        c_ctx2asp_alpha = F.softmax(aspect_x.matmul(self.w_c2a.expand(batch_size, -1, -1)).matmul(x_pool), dim=1)
        c_ctx2asp = torch.matmul(aspect_x.transpose(1, 2), c_ctx2asp_alpha).squeeze(-1)

        x = torch.cat([c_asp2ctx, f_asp2ctx, f_ctx2asp, c_ctx2asp], dim=1)

        return x

    def encoder_mode(self, text_slices, aspect_tokens, **kwargs):
        pos_tuple = kwargs['pos_tuple']
        batch_size = text_slices.size(0)
        x_len = torch.sum(text_slices != 0, dim=1)
        aspect_len = torch.sum(aspect_tokens != 0, dim=1)

        return self.mgan_kernel(text_slices, aspect_tokens, x_len, aspect_len, pos_tuple, batch_size)

    def analysis_mode(self, text_slices, aspect_tokens, x_shape, **kwargs):
        pos_tuple = kwargs['pos_tuple']
        batch_size = text_slices.size(0)
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        aspect_len = torch.sum(aspect_tokens != 0, dim=1)

        return self.mgan_kernel(text_slices, aspect_tokens, x_len, aspect_len, pos_tuple, batch_size)


class MGANHAOFL(PositionHAOFL):
    """A model constructed with HAOFL framework, MGAN is the model used in DPL layer, and LSTM is used in SAL layer."""
    def __init__(self, opt, tokenizer, embedding_matrix):
        super(MGANHAOFL, self).__init__(opt, tokenizer, embedding_matrix, 8)

    def set_dpl(self, embedding_matrix, opt):
        return MGANDPL(embedding_matrix, opt)
