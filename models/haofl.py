# coding: utf-8

import torch
import torch.nn as nn

from layers import DynamicLSTM, SqueezeEmbedding
from models import DTLLayer, DPLLayer, SALLayer


class BaseDPL(DPLLayer):
    def __init__(self, embedding_matrix, opt):
        super(BaseDPL, self).__init__(opt)

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embed = SqueezeEmbedding()
        self.slice_lstm = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)

    def encoder_mode(self, text_slices, aspect_tokens):
        x_len = torch.sum(text_slices != 0, dim=-1)
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1, dtype=torch.float).to(self.opt.device)
        x_len_max = torch.max(x_len)
        text_slices = self.embed(text_slices)
        text_slices = self.squeeze_embed(text_slices, x_len)
        aspect_tokens = self.embed(aspect_tokens)
        aspect_tokens = torch.div(torch.sum(aspect_tokens, dim=1, dtype=torch.float).to(self.opt.device), aspect_len.view(aspect_len.size(0), 1))
        aspect_tokens = torch.unsqueeze(aspect_tokens, dim=1).expand(-1, x_len_max, -1)

        x = torch.cat((text_slices, aspect_tokens), dim=-1).to(self.opt.device)
        _, (x, _) = self.slice_lstm(x, x_len)

        return x[0]

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        aspect_len = torch.sum(aspect_tokens != 0, dim=-1, dtype=torch.float).to(self.opt.device)
        x_len = torch.sum(text_slices[:, (x_shape[1] - 1) * x_shape[2]:] != 0, dim=-1) + (x_shape[1] - 1) * x_shape[2]
        x_len_max = torch.max(x_len)

        text_slices = self.embed(text_slices)
        text_slices = self.squeeze_embed(text_slices, x_len)
        aspect_tokens = self.embed(aspect_tokens)
        aspect_tokens = torch.div(torch.sum(aspect_tokens, dim=1, dtype=torch.float).to(self.opt.device), aspect_len.view(aspect_len.size(0), 1))
        aspect_tokens = torch.unsqueeze(aspect_tokens, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((text_slices, aspect_tokens), dim=-1)

        _, (x, _) = self.slice_lstm(x, x_len)
        return x[0]


class BaseSAL(SALLayer):
    def __init__(self, opt):
        super(BaseSAL, self).__init__(opt)

        self.agg_lstm = DynamicLSTM(opt.hidden_dim, opt.hidden_dim, num_layers=1, batch_first=True)

    def sal_model(self, result_vector, result_len):
        _, (x, _) = self.agg_lstm(result_vector, result_len)

        return x[0]


class BaseHAOFL(nn.Module):
    def __init__(self, opt, tokenizer, embedding_matrix):
        super(BaseHAOFL, self).__init__()

        self.dtl = DTLLayer(opt, tokenizer)
        self.dpl = BaseDPL(embedding_matrix, opt)
        self.sal = BaseSAL(opt)

    def forward(self, trans_method, dpl_mode, train=False, **kwargs):
        inputs = kwargs["inputs"]
        text_slices, aspect_tokens = inputs['text'], inputs['aspect']
        if not train:
            text_slices, aspect_tokens = self.dtl(text_slices, aspect_tokens, trans_method)
        result, group = self.dpl(dpl_mode, text_slices, aspect_tokens, trans_method)
        if dpl_mode == "analysis":
            final = result
        else:
            true_batch_size = len(text_slices)
            slice_num = text_slices.size(1)
            final = self.sal(result, group, true_batch_size, slice_num, dpl_mode)

        return final
