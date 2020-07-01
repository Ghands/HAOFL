# coding:utf-8

import os
import spacy

# The model for splitting sentences in document
spacy_nlp = spacy.load('en_core_web_sm')

# Following lists are used in `TrainDataset`
normal_process_models = ['baseline', 'atae', 'ian', 'aoa', 'bert']
without_aspect_models = ['memnet']
with_position_models = ['ram', 'tnet', 'mgan']

# Log directory
log_dir = './running_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
