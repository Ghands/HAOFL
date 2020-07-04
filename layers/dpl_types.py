# coding:utf-8

import torch
import torch.nn as nn


class DPLLayer(nn.Module):
    def __init__(self, opt):
        """
        The basic Dependency Processing Layer.
        """
        super(DPLLayer, self).__init__()

        self.opt = opt

    def encoder_mode(self, text_slices, aspect_tokens):
        """
        Operations that will be performed when choose `encoder` mode.
        :param text_slices: The input matrix with size [batch_size x text_slice_num, words_in_single_slice]
        :param aspect_tokens: The input matrix of aspect tokens with size [batch_size x text_slices_num, aspect_len]
        :return: Obtained results, every vector means the sentiment representation of a slice.
        """
        raise NotImplementedError("The encoder mode must be implemented according to the logic of used model!")

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        """
        Operations that will be performed when choose `analysis` mode.
        :param text_slices: The input matrix with size [batch_size x text_slice_num, words_in_single_slice]
        :param aspect_tokens: The input matrix of aspect tokens with size [batch_size x text_slices_num, aspect_len]
        :param x_shape: The shape of original input matrix that has not been flatten.
        :return: Obtained results, every vector means the sentiment representation of a document.
        """
        raise NotImplementedError("The analysis mode must be implemented according to the logic of used model!")

    def forward(self, dpl_mode, text_slices, aspect_tokens, trans_method):
        """

        :param dpl_mode: The chosen mode in DPL layer, `analysis` and `encoder` can be chosen. We recommend `encoder`
        :param text_slices: The input matrix with size [batch_size x text_slice_num, words_in_single_slice]
        :param aspect_tokens: The input matrix of aspect tokens with size [batch_size x text_slices_num, aspect_len]
        :param trans_method: The chosen method in DTL layer. If `analysis` is the value of `dpl_mode`,
        `splitting` and `sliding` will not be supported.
        :return: Obtained results, if `encoder` is chosen, `group` will be returned, too.
        `group` indicates the document of each slice.
        """
        if dpl_mode == "encoder":
            true_batch_size = text_slices.size(0)
            x_shape = text_slices.size()
            text_slices = text_slices.flatten(0, 1).long()
            x_len = torch.sum(text_slices != 0, dim=-1)
            group = torch.arange(true_batch_size).unsqueeze(1).expand(true_batch_size, x_shape[1]).flatten().to(
                self.opt.device)
            aspect_size = aspect_tokens.size()
            aspect_tokens = aspect_tokens.expand(x_shape[1], aspect_size[0], aspect_size[1]).permute(1, 0, 2).flatten(0,
                                                                                                                      1)
            text_slices = text_slices[x_len != 0]
            group = group[x_len != 0]
            aspect_tokens = aspect_tokens[x_len != 0]

            result_vector = self.encoder_mode(text_slices, aspect_tokens)
            return result_vector, group
        elif dpl_mode == "analysis":
            if trans_method != "filter":
                raise ValueError("Only the text filter method is supported before DPL when using the analysis mode!")

            x_shape = text_slices.size()
            text_slices = text_slices.flatten(1, 2)

            result_vector = self.analysis_mode(text_slices, aspect_tokens, x_shape)
            return result_vector, None
        else:
            raise ValueError("The value of `dpl_mode` is not supported!")


class PositionDPLLayer(DPLLayer):
    def __init__(self, opt):
        """
        The Dependency Process Layer with position encoding.
        """
        super(PositionDPLLayer, self).__init__(opt)

    def encoder_mode(self, text_slices, aspect_tokens, **kwargs):
        raise NotImplementedError("The encoder mode must be implemented according to the logic of used model!")

    def analysis_mode(self, text_slices, aspect_tokens, x_shape, **kwargs):
        raise NotImplementedError("The analysis mode must be implemented according to the logic of used model!")

    def forward(self, dpl_mode, text_slices, aspect_tokens, trans_method, **kwargs):
        pos_tuple = kwargs["pos_tuple"]
        if dpl_mode == "encoder":
            true_batch_size = text_slices.size(0)
            x_shape = text_slices.size()
            text_slices = text_slices.flatten(0, 1).long()
            pos_tuple = pos_tuple.flatten(0, 1)
            x_len = torch.sum(text_slices != 0, dim=-1)
            group = torch.arange(true_batch_size).unsqueeze(1).expand(true_batch_size, x_shape[1]).flatten().to(
                self.opt.device)
            aspect_size = aspect_tokens.size()
            aspect_tokens = aspect_tokens.expand(x_shape[1], aspect_size[0], aspect_size[1]).permute(1, 0, 2).flatten(0,
                                                                                                                      1)
            text_slices = text_slices[x_len != 0]
            pos_tuple = pos_tuple[x_len != 0]
            group = group[x_len != 0]
            aspect_tokens = aspect_tokens[x_len != 0]

            result_vector = self.encoder_mode(text_slices, aspect_tokens, pos_tuple=pos_tuple)
            return result_vector, group
        elif dpl_mode == "analysis":
            if trans_method != "filter":
                raise ValueError("Only the text filter method is supported before DPL when using the analysis mode!")

            x_shape = text_slices.size()
            text_slices = text_slices.flatten(1, 2)
            pos_tuple = pos_tuple.flatten(1, 2)

            result_vector = self.analysis_mode(text_slices, aspect_tokens, x_shape, pos_tuple=pos_tuple)
            return result_vector, None
        else:
            raise ValueError("The value of `dpl_mode` is not supported!")