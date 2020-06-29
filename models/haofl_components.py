# coding:utf-8

import torch
import numpy as np
import torch.nn as nn

from config import spacy_nlp


class DTLLayer(nn.Module):
    def __init__(self, opt, tokenizer):
        super(DTLLayer, self).__init__()
        
        self.opt = opt
        self.tokenizer = tokenizer
        self.fixed_size = opt.fix_max_len

    def text_preprocessing(self, x_str, **kwargs):
        raise NotImplementedError("You must implement the text preprocessing methods!")

    def single_text_splitting(self, single_str, size, **kwargs):
        words = single_str.split(' ')
        if self.opt.batch:
            words.extend([''] * (self.fixed_size - len(words)))
        text_slices = list()
        for i in range(len(words) // size + 1):
            start = i * size
            text_slices.append(words[start: start + size])
        for idx, sub_words in text_slices:
            text_slices[idx] = self.tokenizer.text_to_sequence(sub_words, maxlen=size + 30)
        return text_slices

    def splitting_window(self, x_str, **kwargs):
        if type(x_str) not in (list, tuple):
            raise TypeError("The input must be LIST type or TUPLE type")

        size = self.opt.split_size

        total_slices = list()
        for single_str in x_str:
            total_slices.append(self.single_text_splitting(single_str, size))

        return np.asarray(total_slices, dtype=np.int64)

    def single_text_slidding(self, single_str, size, stride, **kwargs):
        words = single_str.split(' ')
        if self.opt.batch:
            words.extend([''] * (self.fixed_size - len(words)))
        text_slices = list()
        for i in range((len(words) - size) // stride + 1):
            start = i * stride
            text_slices.append(words[start: start + size])
        for idx, sub_words in text_slices:
            text_slices[idx] = self.tokenizer.text_to_sequence(sub_words, maxlen=size + 30)
        return text_slices

    def sliding_window(self, x_str, **kwargs):
        if type(x_str) not in (list, tuple):
            raise TypeError("The input must be LIST type or TUPLE type")

        size = self.opt.slide_size
        stride = self.opt.stride_size

        total_slices = list()
        for single_str in x_str:
            total_slices.append(self.single_text_slidding(single_str, size, stride))

        return np.asarray(total_slices, dtype=np.int64)

    def single_text_filter(self, single_str, aspect_str, back_pos, forward_pos, text_slice_num, **kwargs):
        all_sentences = [item.text for item in spacy_nlp(single_str).sents]
        aspect_sentences = list()
        if self.opt.batch:
            for idx, content in enumerate(all_sentences):
                if len(aspect_sentences) >= text_slice_num:
                    break
                if aspect_str in content:
                    aspect_sentences.append(
                        self.tokenizer.text_to_sequence(all_sentences[max(0, idx - back_pos): idx + forward_pos]))
            if len(aspect_sentences) < text_slice_num:
                aspect_sentences.extend(
                    [self.tokenizer.text_to_sequence('')] * (text_slice_num - len(aspect_sentences)))
        else:
            for idx, content in enumerate(all_sentences):
                if aspect_str in content:
                    aspect_sentences.append(
                        self.tokenizer.text_to_sequence(all_sentences[max(0, idx - back_pos): idx + forward_pos]))
        return aspect_sentences

    def text_filter(self, x_str, aspect_str, **kwargs):
        if type(x_str) not in (list, tuple):
            raise TypeError("The input must be LIST type or TUPLE type")
        if type(aspect_str) not in (list, tuple):
            raise TypeError("The input must be LIST type or TUPLE type")

        sentence_num = self.opt.sentence_num
        text_slice_num = self.opt.text_slice_num
        back_pos = (sentence_num - 1) // 2
        forward_pos = (sentence_num + 2) // 2

        total_slices = list()
        for single_str, single_aspect in zip(x_str, aspect_str):
            total_slices.append(self.single_text_filter(single_str, single_aspect, back_pos, forward_pos, text_slice_num))

        return np.asarray(total_slices, dtype=np.int64)

    def tokenize_aspect(self, aspect_str):
        aspect_tokens = list()
        for item in aspect_str:
            aspect_tokens.append(self.tokenizer.text_to_sequence(item, maxlen=self.opt.max_aspect_len))
        return np.asarray(aspect_tokens, dtype=np.int64)

    def forward(self, x_str, aspect_str, trans_method):
        assert len(x_str) == len(aspect_str)

        if trans_method == "splitting":
            text_slices = self.splitting_window(x_str)
        elif trans_method == "sliding":
            text_slices = self.sliding_window(x_str)
        elif trans_method == "filter":
            text_slices = self.text_filter(x_str, aspect_str)
        else:
            raise ValueError("The value of `trans_method` is not supported!")

        aspect_tokens = self.tokenize_aspect(aspect_str)

        return text_slices, aspect_tokens


class DPLLayer(nn.Module):
    def __init__(self, opt):
        super(DPLLayer, self).__init__()

        self.opt = opt

    def encoder_mode(self, text_slices, aspect_tokens):
        raise NotImplementedError("The encoder mode must be implemented according to the logic of used model!")

    def analysis_mode(self, text_slices, aspect_tokens, x_shape):
        raise NotImplementedError("The analysis mode must be implemented according to the logic of used model!")

    def forward(self, dpl_mode, text_slices, aspect_tokens, trans_method):
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


class SALLayer(nn.Module):
    def __init__(self, opt):
        super(SALLayer, self).__init__()

        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
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
        return self.dense(hidden_vector)
