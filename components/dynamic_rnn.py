# coding: utf-8

import torch
import torch.nn as nn


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        A wrapper of LSTM network that can handle arbitrary length of input sequence

        :param input_size: The dimension of the input x
        :param hidden_size: The dimension of vectors during computation
        :param num_layers: The number of stacked LSTMs
        :param bias: The `b` in `XW + b`
        :param batch_first: If `True`, the meaning of the first dimension of input should be batch size
        :param dropout: Randomly ignore some elements during computation by the rate, which can reduce over-fitting
        :param bidirectional: If `True`, the structure will consist of a forward LSTM and backward LSTM.
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)
