# coding:utf-8

import torch.nn as nn

from layers import NormalDTLLayer, PositionDTLLayer, FixedSALLayer


class FixedHAOFL(nn.Module):
    def __init__(self, opt, tokenizer, embedding_matrix, fixed_dim):
        super(FixedHAOFL, self).__init__()

        self.dtl = NormalDTLLayer(opt, tokenizer)
        self.dpl = self.set_dpl(embedding_matrix, opt)
        self.sal = FixedSALLayer(opt, fixed_dim)
        self.dense = nn.Linear(fixed_dim, opt.polarities_dim)

    def set_dpl(self, embedding_matrix, opt):
        raise NotImplementedError('The DPL must be implemented if use FixedHAOFL framework!')

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

        return self.dense(final)


class PositionFixedHAOFL(FixedHAOFL):
    def __init__(self, opt, tokenizer, embedding_matrix, fixed_dim):
        super(PositionFixedHAOFL, self).__init__(opt, tokenizer, embedding_matrix, fixed_dim)

        self.dtl = PositionDTLLayer(opt, tokenizer)

    def set_dpl(self, embedding_matrix, opt):
        raise NotImplementedError(
            "The DPL must be implemented with `pos_tuple` parameter if use PositionFixedHAOFL framework!")

    def forward(self, trans_method, dpl_mode, train=False, **kwargs):
        inputs = kwargs['inputs']
        text_slices, aspect_tokens, aspect_positions = inputs['text'], inputs['aspect'], inputs['position']
        if not train:
            text_slices, aspect_positions, aspect_tokens = self.dtl(text_slices, aspect_tokens, trans_method)
        result, group = self.dpl(dpl_mode, text_slices, aspect_tokens, trans_method, pos_tuple=aspect_positions)
        if dpl_mode == "analysis":
            final = result
        else:
            true_batch_size = len(text_slices)
            slice_num = text_slices.size(1)
            final = self.sal(result, group, true_batch_size, slice_num, dpl_mode)

        return self.dense(final)
