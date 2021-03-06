# coding:utf-8

import os
import torch
import pickle
import numpy as np

from config import spacy_nlp, normal_process_models, without_aspect_models, with_position_models
from torch.utils.data import Dataset
from transformers import BertTokenizer
from layers import NormalDTLLayer, NoAspectDTLLayer, PositionDTLLayer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    """
    Obtain a coarse-grain word tokenizer. A pattern surrounded by space will be a token

    :param fnames: The name of file that stores all tokens of a tokenizer
    :param max_seq_len: The default length of sequence that produced by this tokenizer
    :param dat_fname: The file that stores pre-defined tokens
    :return: Tokenizer()
    """
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
    """
    Form a matrix that stores the word vectors of all words in this document-level dataset.

    :param word2idx: The mapping of word and id
    :param embed_dim: The pre-trained vectors of all words, GLOVE is used.
    :param dat_fname: The file stores all word vectors that will be used in this dataset.
    :return: np.array() with shape [-1, 300]
    """
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
    """
    Make all vectors in an array be the fixed length.

    :param sequence: The single vector should be the fixed length.
    :param maxlen: The fixed length
    :param dtype: The data type of initial array.
    :param padding: Pad desirable value before or behind the elements if the length is less than the fixed length.
    `post` means the end.
    :param truncating: Truncate the front or end elements if the length is bigger than the fixed length.
    `post` means the end.
    :param value: The default value
    :return: np.array() with shape (maxlen, )
    """
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
        """
        Coarse-grained tokenizer

        :param max_seq_len: The default length for output sequence
        :param lower: If true, all character will be transformed into lowercase.
        """
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        """
        Build the mapping between words and ids.

        :param text: The raw text.
        :return:
        """
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', maxlen=None):
        """
        Transform the raw text or text list into a fixed length sequence that is full with tokens.

        :param text: The raw text or text list
        :param reverse: If true, all words will be analyzed backward.
        :param padding: Pad desirable value before or behind the elements if the length is less than the fixed length.
        `post` means the end.
        :param truncating: Truncate the front or end elements if the length is bigger than the fixed length.
        `post` means the end.
        :param maxlen: The user-defined length.
        :return: np.array() with shape (maxlen,) or (self.maxlen,)
        """
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
        """
        Build a tokenizer for BERT model.
        The results produced by this tokenizer is different from the results produces by
        normal `Tokenizer` even though the input text is the same.

        :param max_seq_len: The default length of produced sequence
        :param pretrained_bert_name: The name of used BERT model
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', maxlen=None):
        """
        Transform the raw text or text list into a fixed length sequence that is full with tokens.

        :param text: The raw text or text list
        :param reverse: If true, all words will be analyzed backward.
        :param padding: Pad desirable value before or behind the elements if the length is less than the fixed length.
        `post` means the end.
        :param truncating: Truncate the front or end elements if the length is bigger than the fixed length.
        `post` means the end.
        :param maxlen: The user-defined length.
        :return: np.array() with shape (maxlen,) or (self.maxlen,)
        """
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


class TrainDataset(Dataset):
    def __init__(self, file_name, tokenizer, opt, trans_method, dtl_param, name_tail):
        """
        Build a dataset for training.

        :param file_name: The file stores pre-processed results.
        :param tokenizer: Used tokenzier.
        :param opt: A object stores all hyper-parameters.
        :param trans_method: The chosen method of DTL layer. `splitting`, `sliding`, `filter` are all candidates.
        :param dtl_param: A string indicates the parameters of `trans_method`, which will form the storage file name.
        :param name_tail: A user-defined string, which is also used to form the storage file name.
        """
        super(TrainDataset, self).__init__()

        cached_file_name = '{}_{}-{}_{}.cached'.format(file_name, trans_method, dtl_param, name_tail)
        if os.path.exists(cached_file_name):
            all_data = torch.load(cached_file_name)
        else:
            fin = open(file_name, "r")
            all_lines = fin.readlines()
            fin.close()

            all_data = list()

            if opt.model_name in normal_process_models:
                text_processor = NormalDTLLayer(opt, tokenizer)
            elif opt.model_name in without_aspect_models:
                text_processor = NoAspectDTLLayer(opt, tokenizer)
            elif opt.model_name in with_position_models:
                text_processor = PositionDTLLayer(opt, tokenizer)
            else:
                raise ValueError("Unsupported model, please check the variables in file `config.py`!")

            full_x_str = list()
            full_aspect = list()
            full_polarity = list()
            for i in range(0, len(all_lines), 3):
                full_x_str.append(all_lines[i].strip())
                full_aspect.append(all_lines[i + 1].strip())
                full_polarity.append(all_lines[i + 2].strip())

            if opt.model_name in with_position_models:
                text_slices, aspect_positions, aspect_tokens = text_processor(full_x_str, full_aspect, trans_method)
                for single_slices, single_positions, single_aspect, single_polarity in zip(text_slices, aspect_positions, aspect_tokens, full_polarity):
                    single_data = {
                        "text": single_slices,
                        "position": single_positions,
                        "aspect": single_aspect,
                        "polarity": int(single_polarity) + 1
                    }
                    all_data.append(single_data)
            else:
                text_slices, aspect_tokens = text_processor(full_x_str, full_aspect, trans_method)
                for single_slices, single_aspect, single_polarity in zip(text_slices, aspect_tokens, full_polarity):
                    single_data = {
                        "text": single_slices,
                        "aspect": single_aspect,
                        "polarity": int(single_polarity) + 1
                    }
                    all_data.append(single_data)
            torch.save(all_data, cached_file_name)

        self.data = all_data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)