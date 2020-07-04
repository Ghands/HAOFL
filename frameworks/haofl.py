# coding: utf-8

import torch.nn as nn

from layers import NormalDTLLayer, PositionDTLLayer, NormalSALLayer


class HAOFL(nn.Module):
    def __init__(self, opt, tokenizer, embedding_matrix, times_hidden=1):
        """
        The basic HAOFL framework
        :param opt: The instance contains all hyper-parameters
        :param tokenizer: Word tokenizer
        :param embedding_matrix: The word2vec dictionary contains all words used in document-level ABSC dataset.
        :param times_hidden: The number of expanding the size of both the input vector of the final Dense layer
        when adapting 'analysis' mode in DPL and the input vector of the SAL layer network
        """
        super(HAOFL, self).__init__()

        self.dtl = NormalDTLLayer(opt, tokenizer)
        self.dpl = self.set_dpl(embedding_matrix, opt)
        self.sal = NormalSALLayer(opt, times_hidden)
        self.encoder_dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.analysis_dense = nn.Linear(times_hidden * opt.hidden_dim, opt.polarities_dim)

    def set_dpl(self, embedding_matrix, opt):
        raise NotImplementedError("The DPL must be implemented if use HAOFL framework!")

    def forward(self, trans_method, dpl_mode, train=False, **kwargs):
        inputs = kwargs["inputs"]
        text_slices, aspect_tokens = inputs['text'], inputs['aspect']
        if not train:
            text_slices, aspect_tokens = self.dtl(text_slices, aspect_tokens, trans_method)
        result, group = self.dpl(dpl_mode, text_slices, aspect_tokens, trans_method)
        if dpl_mode == "analysis":
            return self.analysis_dense(result)
        else:
            true_batch_size = len(text_slices)
            slice_num = text_slices.size(1)
            final = self.sal(result, group, true_batch_size, slice_num, dpl_mode)
            return self.encoder_dense(final)


class PositionHAOFL(HAOFL):
    def __init__(self, opt, tokenizer, embedding_matrix, times_hidden=1):
        """
        The HAOFL framework with relative position encoding
        :param opt: The instance contains all hyper-parameters
        :param tokenizer: Word tokenizer
        :param embedding_matrix: The word2vec dictionary contains all words used in document-level ABSC dataset.
        :param times_hidden: The number of expanding the size of both the input vector of the final Dense layer
        when adapting 'analysis' mode in DPL and the input vector of the SAL layer network
        """
        super(PositionHAOFL, self).__init__(opt, tokenizer, embedding_matrix, times_hidden)

        self.dtl = PositionDTLLayer(opt, tokenizer)

    def set_dpl(self, embedding_matrix, opt):
        raise NotImplementedError(
            "The DPL must be implemented with `pos_tuple` parameter if use PositionHAOFL framework!")

    def forward(self, trans_method, dpl_mode, train=False, **kwargs):
        inputs = kwargs['inputs']
        text_slices, aspect_tokens, aspect_positions = inputs['text'], inputs['aspect'], inputs['position']
        if not train:
            text_slices, aspect_positions, aspect_tokens = self.dtl(text_slices, aspect_tokens, trans_method)
        result, group = self.dpl(dpl_mode, text_slices, aspect_tokens, trans_method, pos_tuple=aspect_positions)
        if dpl_mode == "analysis":
            return self.analysis_dense(result)
        else:
            true_batch_size = len(text_slices)
            slice_num = text_slices.size(1)
            final = self.sal(result, group, true_batch_size, slice_num, dpl_mode)
            return self.encoder_dense(final)
