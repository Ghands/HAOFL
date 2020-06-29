# coding: utf-8

import copy

import numpy as np

from models import DTLLayer
from config import spacy_nlp


class NormalDTLLayer(DTLLayer):
    def text_preprocessing(self, x_str, **kwargs):
        return x_str


class NoAspectDTLLayer(DTLLayer):
    def text_preprocessing(self, x_str, **kwargs):
        new_x_str = copy.deepcopy(x_str)
        aspect_str = kwargs["aspect_str"]
        assert len(new_x_str) == len(aspect_str)
        for idx in range(len(new_x_str)):
            new_x_str[idx] = new_x_str[idx].replace(aspect_str[idx], '')
        return new_x_str

    def single_text_filter(self, single_str, aspect_str, back_pos, forward_pos, text_slice_num, **kwargs):
        all_sentences = [item.text for item in spacy_nlp(single_str).sents]
        aspect_sentences = list()
        if self.opt.batch:
            for idx, content in enumerate(all_sentences):
                if len(aspect_sentences) >= text_slice_num:
                    break
                if aspect_str in content:
                    aspect_sentences.append(
                        self.tokenizer.text_to_sequence(' '.join(all_sentences[max(0, idx - back_pos): idx + forward_pos]).replace(aspect_str, '')))
            if len(aspect_sentences) < text_slice_num:
                aspect_sentences.extend(
                    [self.tokenizer.text_to_sequence('')] * (text_slice_num - len(aspect_sentences)))
        else:
            for idx, content in enumerate(all_sentences):
                if aspect_str in content:
                    aspect_sentences.append(
                        self.tokenizer.text_to_sequence(' '.join(all_sentences[max(0, idx - back_pos): idx + forward_pos]).replace(aspect_str, '')))
        return aspect_sentences

    def forward(self, x_str, aspect_str, trans_method):
        assert len(x_str) == len(aspect_str)

        new_x_str = self.text_preprocessing(x_str, aspect_str=aspect_str)
        if trans_method == 'splitting':
            text_slices = self.splitting_window(new_x_str)
        elif trans_method == 'sliding':
            text_slices = self.sliding_window(new_x_str)
        elif trans_method == 'filter':
            text_slices = self.text_filter(x_str, aspect_str)
        else:
            raise ValueError("The value of `trans_method` is not supported!")

        aspect_tokens = self.tokenize_aspect(aspect_str)

        return text_slices, aspect_tokens


class PositionDTLLayer(DTLLayer):
    def text_preprocessing(self, x_str, **kwargs):
        return x_str

    def single_text_splitting(self, single_str, size, **kwargs):
        single_aspect = kwargs['single_aspect']
        text_slices = list()
        pos_tuples = list()
        words = single_str.split(' ')
        if self.opt.batch:
            words.extend([''] * (self.fixed_size - len(words)))
        for i in range(len(words) // size + 1):
            start = i * size
            sub_words = words[start: start + size]
            sub_raw = ' '.join(sub_words)
            sub_pos_tuple = list()
            slice_res = sub_raw.find(single_aspect)
            while slice_res != -1:
                t1 = np.sum(self.tokenizer.text_to_sequence(sub_raw[:slice_res], maxlen=size+30) != 0).item()
                t2 = np.sum(self.tokenizer.text_to_sequence(sub_raw[:slice_res + len(single_aspect)], maxlen=size+30) != 0).item()
                sub_pos_tuple.append(t1)
                sub_pos_tuple.append(t2)
                slice_res = sub_raw.find(single_aspect, slice_res + len(single_aspect))
            sub_pos_tuple.append(-1)
            if len(sub_pos_tuple) < self.opt.aspect_pos_len:
                sub_pos_tuple.extend([-1] * (self.opt.aspect_pos_len - len(sub_pos_tuple)))
            else:
                sub_pos_tuple = sub_pos_tuple[:self.opt.aspect_pos_len]
            text_slices.append(self.tokenizer.text_to_sequence(sub_words, maxlen=size + 30))
            pos_tuples.append(sub_pos_tuple)
        return text_slices, pos_tuples

    def splitting_window(self, x_str, **kwargs):
        if type(x_str) not in (list, tuple):
            raise TypeError("The input must be LIST or TUPLE type")
        aspect_str = kwargs["aspect_str"]
        if type(aspect_str) not in (list, tuple):
            raise TypeError("The input must be LIST or TUPLE type")

        size = self.opt.split_size

        total_slices = list()
        total_positions = list()
        for single_x, single_aspect in zip(x_str, aspect_str):
            single_slices, single_positions = self.single_text_splitting(single_x, size, single_aspect=single_aspect)
            total_slices.append(single_slices)
            total_positions.append(single_positions)

        return np.asarray(total_slices, dtype=np.int64), np.asarray(total_positions, dtype=np.int64)


    def single_text_slidding(self, single_str, size, stride, **kwargs):
        single_aspect = kwargs['single_aspect']
        text_slices = list()
        pos_tuples = list()

        words = single_str.split(' ')
        if self.opt.batch:
            words.extend([''] * (self.fixed_size - len(words)))
        for i in range(len(words) // size + 1):
            start = i * stride
            sub_words = words[start: start + size]
            sub_raw = ' '.join(sub_words)
            sub_pos_tuple = list()
            slice_res = sub_raw.find(single_aspect)
            while slice_res != -1:
                t1 = np.sum(self.tokenizer.text_to_sequence(sub_raw[:slice_res], maxlen=size + 30) != 0).item()
                t2 = np.sum(self.tokenizer.text_to_sequence(sub_raw[:slice_res + len(single_aspect)], maxlen= size + 30) != 0).item()
                sub_pos_tuple.append(t1)
                sub_pos_tuple.append(t2)
                slice_res = sub_raw.find(single_aspect, slice_res + len(single_aspect))
            sub_pos_tuple.append(-1)
            if len(sub_pos_tuple) < self.opt.aspect_pos_len:
                sub_pos_tuple.extend([-1] * (self.opt.aspect_pos_len - len(sub_pos_tuple)))
            else:
                sub_pos_tuple = sub_pos_tuple[:self.opt.aspect_pos_len]
            text_slices.append(self.tokenizer.text_to_sequence(sub_words, maxlen=size + 30))
            pos_tuples.append(sub_pos_tuple)
        return text_slices, pos_tuples

    def sliding_window(self, x_str, **kwargs):
        if type(x_str) not in (list, tuple):
            raise TypeError("The input must be LIST or TUPLE type")
        aspect_str = kwargs["aspect_str"]
        if type(aspect_str) not in (list, tuple):
            raise TypeError("The input must be LIST or TUPLE type")

        size = self.opt.slide_size
        stride = self.opt.stride_size

        total_slices = list()
        total_positions = list()
        for single_x, single_aspect in zip(x_str, aspect_str):
            single_slices, single_positions = self.single_text_slidding(single_x, size, stride, single_aspect=single_aspect)
            total_slices.append(single_slices)
            total_positions.append(single_positions)

        return np.asarray(total_slices, dtype=np.int64), np.asarray(total_positions, dtype=np.int64)

    def single_text_filter(self, single_str, aspect_str, back_pos, forward_pos, text_slice_num, **kwargs):
        all_sentences = [item.text for item in spacy_nlp(single_str).sents]
        aspect_sentences = list()
        pos_tuples = list()
        if self.opt.batch:
            for idx, content in enumerate(all_sentences):
                if len(aspect_sentences) >= text_slice_num:
                    break
                if aspect_str in content:
                    selected_sentences = ' '.join(all_sentences[max(0, idx - back_pos): idx + forward_pos])
                    sub_pos_tuple = list()
                    slice_res = selected_sentences.find(aspect_str)
                    while slice_res != -1:
                        t1 = np.sum(self.tokenizer.text_to_sequence(selected_sentences[:slice_res]) != 0).item()
                        t2 = np.sum(self.tokenizer.text_to_sequence(selected_sentences[:slice_res + len(aspect_str)]) != 0).item()
                        sub_pos_tuple.append(t1)
                        sub_pos_tuple.append(t2)
                        slice_res = selected_sentences.find(aspect_str)
                    sub_pos_tuple.append(-1)
                    if len(sub_pos_tuple) < self.opt.aspect_pos_len:
                        sub_pos_tuple.extend([-1] * (self.opt.aspect_pos_len - len(sub_pos_tuple)))
                    else:
                        sub_pos_tuple = sub_pos_tuple[:self.opt.aspect_pos_len]
                    aspect_sentences.append(self.tokenizer.text_to_sequence(selected_sentences))
                    pos_tuples.append(sub_pos_tuple)
            if len(aspect_sentences) < text_slice_num:
                aspect_sentences.extend([self.tokenizer.text_to_sequence('')] * (text_slice_num - len(aspect_sentences)))
                pos_tuples.extend([[-1] * self.opt.aspect_pos_len] * (text_slice_num - len(pos_tuples)))
        else:
            for idx, content in enumerate(all_sentences):
                if aspect_str in content:
                    selected_sentences = ' '.join(all_sentences[max(0, idx - back_pos): idx + forward_pos])
                    sub_pos_tuple = list()
                    slice_res = selected_sentences.find(aspect_str)
                    while slice_res != -1:
                        t1 = np.sum(self.tokenizer.text_to_sequence(selected_sentences[:slice_res]) != 0).item()
                        t2 = np.sum(self.tokenizer.text_to_sequence(selected_sentences[:slice_res + len(aspect_str)]) != 0).item()
                        sub_pos_tuple.append(t1)
                        sub_pos_tuple.append(t2)
                        slice_res = selected_sentences.find(aspect_str)
                    sub_pos_tuple.append(-1)
                    if len(sub_pos_tuple) < self.opt.aspect_pos_len:
                        sub_pos_tuple.extend([-1] * (self.opt.aspect_pos_len - len(sub_pos_tuple)))
                    else:
                        sub_pos_tuple = sub_pos_tuple[:self.opt.aspect_pos_len]
                    aspect_sentences.append(self.tokenizer.text_to_sequence(selected_sentences))
                    pos_tuples.append(sub_pos_tuple)
        return aspect_sentences, pos_tuples

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
        total_positions = list()
        for single_x, single_aspect in zip(x_str, aspect_str):
            single_slices, single_positions = self.single_text_filter(single_x, single_aspect, back_pos, forward_pos, text_slice_num)
            total_slices.append(single_slices)
            total_positions.append(single_positions)

        return np.asarray(total_slices, dtype=np.int64), np.asarray(total_positions, dtype=np.int64)

    def forward(self, x_str, aspect_str, trans_method):
        assert len(x_str) == len(aspect_str)

        if trans_method == "splitting":
            text_slices, aspect_positions  = self.splitting_window(x_str)
        elif trans_method == "sliding":
            text_slices, aspect_positions  = self.sliding_window(x_str)
        elif trans_method == "filter":
            text_slices, aspect_positions = self.text_filter(x_str, aspect_str)
        else:
            raise ValueError("The value of `trans_method` is not supported!")

        aspect_tokens = self.tokenize_aspect(aspect_str)

        return text_slices, aspect_positions, aspect_tokens
