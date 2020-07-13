# HAOFL

## Introduction

Hierarchical Aspect-Oriented Framework for Long document (HAOFL) is a framework 
developed for document-level Aspect Based Sentiment Classification (ABSC) task.

Previous ABSC tasks are all sentence-level, which means the analyzed contents that
contain several aspects are short texts with single-digit sentences. These contents
are similar to reviews in shopping platforms and tweets. However, long news and long
blogs are much different from these sentence-level contents. What is the performance
of existing sentence-level ABSC models on these long documents?

We focus on processing long dependencies and aggregating complex sentiment to propose
a framework, HAOFL. This framework has three functional layers to play important roles
in solving document-class ASBC task.

![The structure of HAOFL](./img/HAOFL%20STRUCTURE.png)

1. Data Transformation Layer (DTL) is a layer pre-processes the input long document.
There are three data transformation methods in DTL, splitting window, sliding window,
and text filter. Splitting window and sliding window will obtain little text slices by
the number of words, while text filter can locate target aspects and extract neighbor
sentences.
2. Dependency Processing Layer (DPL) can analyze the sentiment dependency in a text
slice. There are two modes in this layer, encoder mode and analysis mode. Encoder mode
analyzes the single text slice and generates the sentiment representation of this text
slice. Analysis mode studies the sentiment information of all text slices and obtains
the overall sentiment representation of the document. Notably, analysis mode can be 
chosen only when text filter is the data transformation method in DTL. The model used
in DPL can be sentence-level ABSC models, but these models should adjust for some 
differences between documents and sentences.
3. Sentiment Aggregation Layer (SAL) can aggregate the local sentiment representation 
of text slices and produce a global sentiment representation of the whole document.
If the mode of DPL is analysis, SAL will do nothing because the result of analysis mode
is already the global sentiment representation of the whole document.

## Requirements

HAOFL is implemented with Python3, other packages are listed:

- pytorch == 1.4.0
- numpy == 1.18.2
- ransformers == 2.9.1
- spacy == 2.2.4

## How to Use?

You should download the pre-trained model used by the spaCy.

```shell
python3 -m spacy download en_core_web_sm
```

Then, you can running model to perform the document-level ABSC task with HAOFL based 
models.

```shell
python3 train.py --model_name <name of the model>
```

We have implemented 9 types of HAOFL based model, except for the baseline, we also 
integrated sentence-level ABSC models, ATAE, IAN, Memory Network, AOA, RAM, TNET, 
MGAN, and BERT. Running commands for these models are the following.

- Baseline

  ```shell
  python3 train.py --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice>
  ```

- ATAE

  ```shell
  python3 train.py --model_name atae --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice>
  ```

- IAN

  ```shell
  python3 train.py --model_name ian --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice>
  ```

- MemNet

  ```shell
  python3 train.py --model_name memnet --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice> --name_tail NoAspectInText
  ```

- AOA

  ```shell
  python3 train.py --model_name aoa --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice>
  ```

- RAM

  ```shell
  python3 train.py --model_name ram --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice> --name_tail position
  ```

- TNET

  ```shell
  python3 train.py --model_name tnet --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice> --name_tail position
  ```

- MGAN

  ```shell
  python3 train.py --model_name mgan --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice> --name_tail position
  ```

- BERT

  ```shell
  python3 train.py --model_name bert --device cuda:1 --dtl_method <method_name> --dpl_mode <mode_choice> --embed_dim 768 --hidden_dim 768 --batch_size 1 --learning_rate 0.00001 
  ```

Notably, the `name_tail` is a tag for a self-defined data pre-processing method. For 
example, RAM, TNET, and MGAN all adopt relative position encoding methods, so we 
implement a subclass for DTL layer, and the `name_tail` is `position`. When BERT is 
integrated, the GPU RAM is out of the bound, so we assign the `batch_size` to 1.

The running situation of these models are shown in following table.

| Model Name | Splitting Window   | Sliding Window     | Text Filter        | Analysis Mode      |
| ---------- | ------------------ | ------------------ | ------------------ | ------------------ |
| baseline   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| atae       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| ian        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| memnet     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| aoa        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| ram        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| tnet       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| mgan       | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| bert       | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                |

The meaning of the latter four columns are:

- Splitting Window: "splitting window" method in DTL layer and "encoder" mode in 
DPL layer.
- Sliding Window: "sliding window" method in DTL layer and "encoder" mode in DPL 
layer.
- Text Filter: "text filter" method in DTL layer and "encoder" mode in DPL layer.
- Analysis Mode: "text filter" method in DTL layer and "analysis" mode in DPL layer.

There are two failures: 

1. BERT integrated model with "sliding window" method in DTL is out of GPU Memory
2. The input of BERT integrated model with "analysis" mode in DPL is too long, which 
exceeds 512.

Among all methods, we recommend the approach "text filter" method with "encoder",
which not only runs faster but also performs better.

## Document-level Dataset: AOLTD

Aspect-Oriented Long Text Dataset is a document-level dataset contains the contexts 
we collected. Corresponding aspects and polarities are obtained by Guoxin Cloud
Service Co., Ltd. manually. Some information about AOLTD is show in following table.

| Set Type           | Positive | Neutral | Negative |
| ------------------ | -------- | ------- | -------- |
| Training(full)     | 799      | 1597    | 1597     |
| Test(full)         | 234      | 817     | 652      |
| Training(released) | 200      | 400     | 400      |
| Test(released)     | 100      | 200     | 200      |

The shortest content in AOLTD is about 140 words, which is longer than the longest
content in normal sentence-level datasets, e.g. Semeval-14 laptop, Semeval-14 
restaurant, and Twitter.

For some special reasons, we release a part of AOLTD now. The full version of AOLTD
will be released in the near future. 

---

Questions are welcome in 'Issue' module of this repository.