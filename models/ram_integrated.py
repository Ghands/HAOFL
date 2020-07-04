# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from components import DynamicLSTM
from layers import PositionDPLLayer
from frameworks import PositionHAOFL


class RAMDPL(PositionDPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(RAMDPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True,
                                           bidirectional=True)
        self.att_linear = nn.Linear(opt.hidden_dim * 2 + 1 + opt.embed_dim * 2, 1)
        self.gru_cell = nn.GRUCell(opt.hidden_dim * 2 + 1, opt.hidden_dim)

    def location_memory(self, x, x_len, all_pos_tuple):
        batch_size = x.size(0)
        seq_len = x.size(1)
        all_pos_tuple = all_pos_tuple.cpu().numpy()
        x_len = x_len.cpu().numpy()
        weight = list()
        u = list()

        for i in range(batch_size):
            total_len = x_len[i]
            weight_value = list()
            u_value = list()
            pos_tuple = all_pos_tuple[i]
            for j in range(0, len(pos_tuple), 2):
                if pos_tuple[j + 1] > seq_len:
                    continue
                if pos_tuple[j] != -1:
                    weight_value.append(np.concatenate(
                        (1 - np.arange(pos_tuple[j], 0, -1) / total_len, np.ones(pos_tuple[j + 1] - pos_tuple[j]),
                         1 - np.arange(1, total_len - pos_tuple[j + 1] + 1) / total_len,
                         np.ones(seq_len - total_len))))
                    u_value.append(
                        np.concatenate((np.arange(-pos_tuple[j], 0), np.zeros(pos_tuple[j + 1] - pos_tuple[j]),
                                        np.arange(1, total_len - pos_tuple[j + 1] + 1), np.zeros(seq_len - total_len))))
                else:
                    weight_value.append(np.ones(seq_len))
                    u_value.append(np.ones(seq_len))

            weight_value = np.stack(weight_value).max(axis=0)
            u_value = np.stack(u_value).max(axis=0)

            weight.append(weight_value)
            u.append(u_value)
        u = torch.tensor(u, dtype=torch.float).to(self.opt.device).unsqueeze(2)
        weight = torch.tensor(weight, dtype=torch.float).to(self.opt.device).unsqueeze(2)
        x = x * weight
        x = torch.cat([x, u], dim=2)

        return x

    def ram_kernel(self, x, aspect_x, x_len, aspect_len, pos_tuple):
        x = self.embed(x)
        x, (_, _) = self.bi_lstm_context(x, x_len)
        x = self.location_memory(x, x_len, pos_tuple)

        aspect_x = self.embed(aspect_x)
        aspect_x = torch.div(torch.sum(aspect_x, dim=1, dtype=torch.float).to(self.opt.device),
                             aspect_len.float().unsqueeze(-1))
        et = torch.zeros_like(aspect_x).to(self.opt.device)

        batch_size = x.size(0)
        seq_len = x.size(1)
        for _ in range(self.opt.hops):
            g = self.att_linear(torch.cat(
                [x, torch.zeros(batch_size, seq_len, self.opt.embed_dim).to(self.opt.device) + et.unsqueeze(1),
                 torch.zeros(batch_size, seq_len, self.opt.embed_dim).to(self.opt.device) + aspect_x.unsqueeze(1)],
                dim=-1))
            alpha = F.softmax(g, dim=1)
            i = torch.bmm(alpha.transpose(1, 2), x).squeeze(1)
            et = self.gru_cell(i, et)

        return et

    def encoder_mode(self, text_slices, aspect_tokens, **kwargs):
        pos_tuple = kwargs['pos_tuple']
        x_len = torch.sum(text_slices != 0, dim=-1)
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1)

        return self.ram_kernel(text_slices, aspect_tokens, x_len, aspect_len, pos_tuple)

    def analysis_mode(self, text_slices, aspect_tokens, x_shape, **kwargs):
        pos_tuple = kwargs['pos_tuple']
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1)

        return self.ram_kernel(text_slices, aspect_tokens, x_len, aspect_len, pos_tuple)


class RAMHAOFL(PositionHAOFL):
    """A model constructed with HAOFL framework, RAM is the model used in DPL layer, and LSTM is used in SAL layer."""
    def __init__(self, opt, tokenizer, embedding_matrix):
        super(RAMHAOFL, self).__init__(opt, tokenizer, embedding_matrix, 1)

    def set_dpl(self, embedding_matrix, opt):
        return RAMDPL(embedding_matrix, opt)
