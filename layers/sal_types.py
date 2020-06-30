# coding:utf-8

import torch
import torch.nn as nn

from components import DynamicLSTM


class SALLayer(nn.Module):
    def __init__(self, opt):
        super(SALLayer, self).__init__()

        self.opt = opt

    def sal_model(self, result_vector, result_len):
        raise NotImplementedError("The SAL must be implemented!")

    def forward(self, result_vector, group, true_batch_size, slice_num, dpl_mode):
        if dpl_mode != "encoder":
            raise ValueError("The mode of previous DPL must be ENCODER")

        temp_list = list()
        len_list = list()
        for i in range(true_batch_size):
            temp_x = result_vector[group == i]
            len_list.append(temp_x.size(0))
            temp_x = nn.functional.pad(temp_x, (0, 0, 0, slice_num - temp_x.size(0)))
            temp_list.append(temp_x)
        x = torch.stack(temp_list, dim=0).to(self.opt.device)
        x_len = torch.Tensor(len_list).long().to(self.opt.device)

        hidden_vector = self.sal_model(x, x_len)
        return hidden_vector


class NormalSALLayer(SALLayer):
    def __init__(self, opt, times_hidden=1):
        super(NormalSALLayer, self).__init__(opt)

        self.agg_lstm = DynamicLSTM(times_hidden * opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True)

    def sal_model(self, result_vector, result_len):
        _, (x, _) = self.agg_lstm(result_vector, result_len)

        return x[0]


class FixedSALLayer(SALLayer):
    def __init__(self, opt, in_fixed_dim, out_fixed_dim):
        super(FixedSALLayer, self).__init__(opt)

        self.agg_lstm = DynamicLSTM(in_fixed_dim, out_fixed_dim, num_layers=1, batch_first=True)

    def sal_model(self, result_vector, result_len):
        _, (x, _) = self.agg_lstm(result_vector, result_len)

        return x[0]
