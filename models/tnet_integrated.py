# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from components import DynamicLSTM
from layers import PositionDPLLayer
from frameworks import PositionFixedHAOFL


class TNETDPL(PositionDPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(TNETDPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

        self.lstm1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.convs3 = nn.Conv1d(2 * opt.hidden_dim, 50, 3, padding=1)
        self.fc1 = nn.Linear(4 * opt.hidden_dim, 2 * opt.hidden_dim)

    def location_memory(self, x, x_len, all_pos_tuple):
        batch_size = x.size(0)
        seq_len = x.size(1)
        all_pos_tuple = all_pos_tuple.cpu().numpy()
        x_len = x_len.cpu().numpy()
        weight = list()

        for i in range(batch_size):
            total_len = x_len[i]
            weight_value = list()
            pos_tuple = all_pos_tuple[i]
            for j in range(0, len(pos_tuple), 2):
                if pos_tuple[j + 1] > seq_len:
                    continue
                if pos_tuple[j] != -1:
                    weight_value.append(np.concatenate((1 - np.arange(pos_tuple[j + 1], 0, -1) / total_len,
                                                        1 - np.arange(pos_tuple[j + 1] - pos_tuple[j],
                                                                      total_len - pos_tuple[j]) / total_len,
                                                        np.zeros(seq_len - total_len))))
                else:
                    weight_value.append(np.zeros(seq_len))
            weight_value = np.stack(weight_value).max(axis=0)

            weight.append(weight_value)

        x = torch.tensor(weight, dtype=torch.float).to(self.opt.device).unsqueeze(2) * x

        return x

    def tnet_kernel(self, x, aspect_x, x_len, aspect_len, pos_tuple):
        x = self.embed(x)
        aspect_x = self.embed(aspect_x)
        x, (_, _) = self.lstm1(x, x_len)
        aspect_x, (_, _) = self.lstm2(aspect_x, aspect_len)
        x = x.transpose(1, 2)
        aspect_x = aspect_x.transpose(1, 2)

        for i in range(self.opt.hops):
            a = torch.bmm(aspect_x.transpose(1, 2), x)
            a = F.softmax(a, 1)
            aspect_mid = torch.bmm(aspect_x, a)
            aspect_mid = torch.cat((aspect_mid, x), dim=1).transpose(1, 2)
            aspect_mid = F.relu(self.fc1(aspect_mid).transpose(1, 2))
            x = aspect_mid + x
            x = self.location_memory(x.transpose(1, 2), x_len, pos_tuple).transpose(1, 2)

        x = F.relu(self.convs3(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def encoder_mode(self, text_slices, aspect_tokens, **kwargs):
        pos_tuple = kwargs['pos_tuple']
        x_len = torch.sum(text_slices != 0, dim=-1)
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1).to(self.opt.device)

        return self.tnet_kernel(text_slices, aspect_tokens, x_len, aspect_len, pos_tuple=pos_tuple)

    def analysis_mode(self, text_slices, aspect_tokens, x_shape, **kwargs):
        pos_tuple = kwargs['pos_tuple']
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1).to(self.opt.device)

        return self.tnet_kernel(text_slices, aspect_tokens, x_len, aspect_len, pos_tuple=pos_tuple)


class TNETHAOFL(PositionFixedHAOFL):
    def __init__(self, opt, tokenizer, embedding_matrix, fixed_dim):
        super(TNETHAOFL, self).__init__(opt, tokenizer, embedding_matrix, fixed_dim)

    def set_dpl(self, embedding_matrix, opt):
        return TNETDPL(embedding_matrix, opt)
