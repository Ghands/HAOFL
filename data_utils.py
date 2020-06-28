# coding:utf-8

import os
import torch
import spacy
import pickle
import numpy as np

from torch.utils.data import Dataset
from transformers import BertTokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', maxlen=None):
        if maxlen is None:
            maxlen = self.max_seq_len
        if type(text) == list:
            text = " ".join(text)
        if self.lower:
            text = text.lower()

        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, maxlen, padding=padding, truncating=truncating)


class Tokenizer4Bert(object):
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', maxlen=None):
        if maxlen is None:
            maxlen = self.max_seq_len
        if type(text) == list:
            text = " ".join(text)

        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, maxlen, padding=padding, truncating=truncating)


class LongDataset(Dataset):
    def __init__(self, file_name, tokenizer, bert_tokenizer, opt, dataset_tail):
        super(LongDataset, self).__init__()

        window_size = opt.window_size
        back_pos = (window_size - 1) // 2
        forward_pos = (window_size + 2) // 2
        window_num = opt.window_num

        if len(dataset_tail) == 0:
            cached_file_name = "{}.cached".format(file_name)
        else:
            cached_file_name = "{}_{}.cached".format(file_name, dataset_tail)
        if os.path.exists(cached_file_name):
            all_data = torch.load(cached_file_name)
        else:
            fin = open(file_name, "r")
            all_lines = fin.readlines()
            fin.close()

            all_data = list()
            nlp = spacy.load('en_core_web_sm')

            for line_num in range(0, len(all_lines), 3):
                text = all_lines[line_num].strip()
                aspect = all_lines[line_num + 1].strip()
                polarity = all_lines[line_num + 2].strip()

                sample_text = list()
                bert_text = list()
                sub_raw_text = list()
                no_aspect_sample_text = list()
                all_sentences = [item.text for item in nlp(text).sents]
                for idx, context in enumerate(all_sentences):
                    if len(sample_text) > window_num:
                        break
                    if aspect in context:
                        sample_text.append(tokenizer.text_to_sequence(
                            all_sentences[max(0, idx - back_pos): idx + forward_pos]))
                        no_aspect_sample_text.append(tokenizer.text_to_sequence(
                            [temp_sentence.replace(aspect, "") for temp_sentence in
                             all_sentences[max(0, idx - back_pos): idx + forward_pos]]))
                        bert_text.append(bert_tokenizer.text_to_sequence(
                            all_sentences[max(0, idx - back_pos): idx + forward_pos], maxlen=opt.max_seq_len))
                        sub_raw_text.append(" ".join(all_sentences[max(0, idx - back_pos): idx + forward_pos]))
                assert len(sample_text) == len(bert_text)
                if len(sample_text) < window_num:
                    now_length = len(sample_text)
                    sample_text.extend([tokenizer.text_to_sequence('')] * (window_num - now_length))
                    no_aspect_sample_text.extend([tokenizer.text_to_sequence('')] * (window_num - now_length))
                    bert_text.extend(
                        [bert_tokenizer.text_to_sequence('', maxlen=opt.max_seq_len)] * (window_num - now_length))
                elif len(sample_text) > window_num:
                    sample_text = sample_text[:window_num]
                    no_aspect_sample_text = no_aspect_sample_text[:window_num]
                    bert_text = bert_text[:window_num]
                    sub_raw_text = sub_raw_text[:window_num]
                assert len(sample_text) == len(bert_text)
                assert len(sample_text) == len(no_aspect_sample_text)
                sample_text = np.asarray(sample_text, dtype=np.int64)
                no_aspect_sample_text = np.asarray(no_aspect_sample_text, dtype=np.int64)

                # bert_text_indices = bert_tokenizer.text_to_sequence(text)
                bert_aspect_indices = bert_tokenizer.text_to_sequence(aspect, maxlen=20)
                bert_window_indices = np.asarray(bert_text, dtype=np.int64)

                raw_text_indices = tokenizer.text_to_sequence(text, maxlen=opt.max_bert_len)
                raw_text_no_aspect_indices = tokenizer.text_to_sequence(text.replace(aspect, ""), maxlen=opt.max_bert_len)

                position_tuple_list = []
                find_res = text.find(aspect)
                while find_res != -1:
                    t1 = np.sum(tokenizer.text_to_sequence(text[:find_res], maxlen=opt.max_bert_len) != 0).item()
                    t2 = np.sum(tokenizer.text_to_sequence(text[:find_res + len(aspect)], maxlen=opt.max_bert_len) != 0).item()
                    position_tuple_list.append(t1)
                    position_tuple_list.append(t2)
                    find_res = text.find(aspect, find_res + len(aspect))
                position_tuple_list.append(-1)
                if len(position_tuple_list) < opt.aspect_pos_len:
                    position_tuple_list.extend([-1] * (opt.aspect_pos_len - len(position_tuple_list)))
                else:
                    position_tuple_list = position_tuple_list[:opt.aspect_pos_len]
                position_tuple_list = np.asarray(position_tuple_list, dtype=np.int64)

                part_pos_tuple_list = []
                for idx, sub_raw in enumerate(sub_raw_text):
                    part_res = sub_raw.find(aspect)
                    while part_res != -1:
                        t1 = np.sum(tokenizer.text_to_sequence(sub_raw[:part_res]) != 0).item()
                        t2 = np.sum(tokenizer.text_to_sequence(sub_raw[:part_res + len(aspect)]) != 0).item()
                        part_pos_tuple_list.append(t1 + idx * tokenizer.max_seq_len)
                        part_pos_tuple_list.append(t2 + idx * tokenizer.max_seq_len)
                        part_res = sub_raw.find(aspect, part_res + len(aspect))
                part_pos_tuple_list.append(-1)
                if len(part_pos_tuple_list) < opt.aspect_pos_len:
                    part_pos_tuple_list.extend([-1] * (opt.aspect_pos_len - len(part_pos_tuple_list)))
                else:
                    part_pos_tuple_list = part_pos_tuple_list[:opt.aspect_pos_len]
                part_pos_tuple_list = np.asarray(part_pos_tuple_list, dtype=np.int64)

                shared_pos_tuple_list = []
                for idx, sub_raw in enumerate(sub_raw_text):
                    share_res = sub_raw.find(aspect)
                    sub_pos_list = []
                    while share_res != -1:
                        t1 = np.sum(tokenizer.text_to_sequence(sub_raw[:share_res]) != 0).item()
                        t2 = np.sum(tokenizer.text_to_sequence(sub_raw[:share_res + len(aspect)]) != 0).item()
                        sub_pos_list.append(t1)
                        sub_pos_list.append(t2)
                        share_res = sub_raw.find(aspect, share_res + len(aspect))
                    sub_pos_list.append(-1)
                    if len(sub_pos_list) < opt.aspect_pos_len:
                        sub_pos_list.extend([-1] * (opt.aspect_pos_len - len(sub_pos_list)))
                    else:
                        sub_pos_list = sub_pos_list[:opt.aspect_pos_len]
                    shared_pos_tuple_list.append(sub_pos_list)
                if len(shared_pos_tuple_list) < opt.window_num:
                    shared_pos_tuple_list.extend([[-1] * opt.aspect_pos_len for _ in range(0, opt.window_num - len(shared_pos_tuple_list))])
                else:
                    shared_pos_tuple_list = shared_pos_tuple_list[:opt.window_num]
                shared_pos_tuple_list = np.asarray(shared_pos_tuple_list, dtype=np.int64)

                total_slice_num = (opt.max_bert_len - opt.slice_size) // opt.slice_stride + 1
                text_words = text.split(" ")
                text_slices = list()
                for i in range(total_slice_num):
                    start = i * opt.slice_stride
                    text_slices.append(text_words[start: start + opt.slice_size])
                slice_pos_tuple_list = list()
                slice_text_tuple_list = list()
                bert_text_tuple_list = list()
                for idx, sub_words in enumerate(text_slices):
                    sub_pos_list = []
                    if len(sub_words) == 0:
                        sub_pos_list.extend([-1] * (opt.aspect_pos_len - len(sub_pos_list)))
                        sub_text_list = tokenizer.text_to_sequence("", maxlen=opt.slice_size + 30)
                        sub_bert_list = bert_tokenizer.text_to_sequence("", maxlen=opt.slice_size + 50)
                    else:
                        sub_raw = " ".join(sub_words)
                        slice_res = sub_raw.find(aspect)
                        while slice_res != -1:
                            t1 = np.sum(tokenizer.text_to_sequence(sub_raw[:slice_res], maxlen=opt.slice_size + 30) != 0).item()
                            t2 = np.sum(tokenizer.text_to_sequence(sub_raw[:slice_res + len(aspect)], maxlen=opt.slice_size + 30) != 0).item()
                            sub_pos_list.append(t1)
                            sub_pos_list.append(t2)
                            slice_res = sub_raw.find(aspect, slice_res + len(aspect))
                        sub_pos_list.append(-1)
                        if len(sub_pos_list) < opt.aspect_pos_len:
                            sub_pos_list.extend([-1] * (opt.aspect_pos_len - len(sub_pos_list)))
                        else:
                            sub_pos_list = sub_pos_list[:opt.aspect_pos_len]
                        sub_text_list = tokenizer.text_to_sequence(sub_raw, maxlen=opt.slice_size + 30)
                        sub_bert_list = bert_tokenizer.text_to_sequence(sub_raw, maxlen=opt.slice_size + 50)
                    slice_pos_tuple_list.append(sub_pos_list)
                    slice_text_tuple_list.append(sub_text_list)
                    bert_text_tuple_list.append(sub_bert_list)
                slice_pos_tuple_list = np.asarray(slice_pos_tuple_list, dtype=np.int64)
                slice_text_tuple_list = np.asarray(slice_text_tuple_list, dtype=np.int64)
                bert_text_tuple_list = np.asarray(bert_text_tuple_list, dtype=np.int64)

                data = {
                    "context_window_indices": sample_text,
                    "aspect_indices": tokenizer.text_to_sequence(aspect),
                    "polarity": int(polarity) + 1,
                    'bert_text_indices': bert_text_tuple_list,
                    'bert_aspect_indices': bert_aspect_indices,
                    'bert_window_indices': bert_window_indices,
                    "no_aspect_context_window_indices": no_aspect_sample_text,
                    "raw_text_indices": raw_text_indices,
                    "raw_text_no_aspect_indices": raw_text_no_aspect_indices,
                    'full_position_tuple_list': position_tuple_list,
                    'position_tuple_list': part_pos_tuple_list,
                    "shared_pos_tuple_list": shared_pos_tuple_list,
                    'slice_position_tuple_list': slice_pos_tuple_list,
                    'slice_text_tuple_list': slice_text_tuple_list
                }

                all_data.append(data)

            torch.save(all_data, cached_file_name)

        self.data = all_data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
