#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date   : 2021/3/31 13:50
@Author : qiangzi
@File   : pytorch_text_2.py
@Todo   : pytorch_text_2
"""
import os
import io
import logging

from collections import Counter
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader

# from torchtext.datasets import AG_NEWS
from torchtext.utils import download_from_url
from torchtext.utils import extract_archive
from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import Vocab
# from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets.text_classification import TextClassificationDataset
# from torchtext.datasets.text_classification import URLS, _csv_iterator, _create_data_from_iterator
from torchtext.datasets.text_classification import URLS


# __all__ = []

# __log_format = "%(asctime)s [ %(levelname)-7s ] | %(filename)-24s(line:%(lineno)-4s)-%(process)d(%(thread)d) || %(message)s"
__log_format = "%(asctime)s [ %(levelname)-7s ] | %(process)d || %(message)s"
# __date_format = "%Y-%m-%d(%A) %H:%M:%S(%Z)"
__date_format = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=__log_format, datefmt=__date_format)
logger = logging.getLogger(__name__)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 选择分词器
tokenizer = get_tokenizer("basic_english")


def csv_iterator(data_path, ngrams, yield_cls=False):
    """
    加载csv文本文件，并根据原始文本 生成 指定ngram语法的 词汇(token)样本
    Args:
        data_path:
        ngrams:
        yield_cls:

    Returns:

    """
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)


def create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                        for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


def build_vocab_from_iterator(iterator, num_lines=None):
    """
    Build a Vocab from an iterator.

    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        num_lines: The expected number of elements returned by the iterator.
            (Default: None)
            Optionally, if known, the expected number of elements can be passed to
            this factory function for improved progress reporting.
    """

    counter = Counter()
    with tqdm(unit_scale=0, unit='lines', total=num_lines) as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)
    word_vocab = Vocab(counter)
    return word_vocab


def _setup_datasets(dataset_name, root='.data', ngrams=1, vocab=None, include_unk=False):
    ##################################################################################
    # 由于torchtext实验数据集的 原有逻辑 会去下载 国外的墙外数据集，
    # 所以在自行下载实验数据集后，调整了已下载数据集的加载逻辑。
    # 已下载数据集AG_NEWS 和 SogouNews，数据集存放地址: ~/data/torch
    data_dir = os.path.join(os.path.expanduser('~'), "data/torch")
    dataset_file_name = {
        'AG_NEWS': os.path.join(data_dir, "ag_news_csv.tar.gz"),
        'SogouNews': os.path.join(data_dir, "sogou_news_csv.tar.gz"),
    }

    extracted_files = []
    if dataset_name in dataset_file_name.keys():
        extracted_files = extract_archive(dataset_file_name[dataset_name])
        pass
    else:
        dataset_tar = download_from_url(URLS[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)

    train_csv_path = ""
    test_csv_path = ""
    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname
    ##################################################################################

    # 数据文件加载
    # 构建词汇表
    logger.info("Creating Vocab")
    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    # 构建训练数据集
    logging.info('Creating training data')
    train_data, train_labels = create_data_from_iterator(vocab, csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    logger.info(f"training data size: {len(train_data)}")
    # 构建猜测是数据集
    logging.info('Creating testing data')
    test_data, test_labels = create_data_from_iterator(vocab, csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    logger.info(f"testing data size: {len(test_data)}")
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (
        TextClassificationDataset(vocab, train_data, train_labels),
        TextClassificationDataset(vocab, test_data, test_labels)
    )


def AG_NEWS(*args, **kwargs):
    """ Defines AG_NEWS datasets.
        The labels includes:
            - 0 : World
            - 1 : Sports
            - 2 : Business
            - 3 : Sci/Tech
    Create supervised learning dataset: AG_NEWS
    Separately returns the training and test dataset
    Arguments:
        # root: Directory where the datasets are saved. Default: ".data"
        # ngrams: a contiguous sequence of n items from s string text.
        #     Default: 1
        # vocab: Vocabulary used for dataset. If None, it will generate a new
        #     vocabulary based on the train data set.
        # include_unk: include unknown token in the data (Default: False)
    Examples:
        # >>> train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)
    """

    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


ag_news_label = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tec"
}


def SogouNews(*args, **kwargs):
    """ Defines SogouNews datasets.
        The labels includes:
            - 0 : Sports
            - 1 : Finance
            - 2 : Entertainment
            - 3 : Automobile
            - 4 : Technology
    Create supervised learning dataset: SogouNews
    Separately returns the training and test dataset
    Arguments:
        # root: Directory where the datasets are saved. Default: ".data"
        # ngrams: a contiguous sequence of n items from s string text.
        #     Default: 1
        # vocab: Vocabulary used for dataset. If None, it will generate a new
        #     vocabulary based on the train data set.
        # include_unk: include unknown token in the data (Default: False)
    Examples:
        # >>> train_dataset, test_dataset = torchtext.datasets.SogouNews(ngrams=3)
    """

    return _setup_datasets(*(("SogouNews",) + args), **kwargs)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(_label)
        text_list.append(_text)
        offsets.append(_text.size()[0])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1])
    offsets = offsets.cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_class = num_class
        # 初始化网络结构
        self.embedding = nn.EmbeddingBag(self.vocab_size, self.embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        # 初始化网络参数
        self.init_weights()

    def init_weights(self):
        """
        网络权重初始化
        Returns:
        """
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        pass

    def forward(self, text, offsets):
        """
        前向计算
        Returns:
        """
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    pass


def training(model, samples_dataloader, epochs):
    # 优化设置
    criterion = nn.CrossEntropyLoss().to(device)
    learning_rate = 4
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    logger.info(f"trainning setting: {criterion}")
    logger.info(f"trainning setting: {optimizer}")
    logger.info(f"trainning setting: {scheduler}")
    # 训练迭代
    logger.info("training starting ...")
    for epoch in range(epochs):
        training_loss = 0
        training_acc = 0
        logger.info(f"==========>>Epoch {epoch}<<==========")
        # 迭代batch
        num_batch = 1
        for idx, (labels, samples, offsets) in enumerate(samples_dataloader):
            optimizer.zero_grad()
            labels, samples, offsets = labels.to(device), samples.to(device), offsets.to(device)
            # 前向计算
            output = model.to(device)(samples, offsets)
            # 误差计算
            loss = criterion(output, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 记录
            acc = (output.argmax(1) == labels).sum().item() / output.size()[0]
            training_acc += acc
            training_loss += loss.item()
            if 0 == (idx % 100):
                logger.info(f"batch: {idx: >6d}, train loss: {loss: >8.4f}, train acc:  {acc: >8.4f}")
                pass
            num_batch += 1
            pass
        scheduler.step()  # 学习率更新
        logger.info(f"Epoch end: train loss: {training_loss / num_batch: >8.4f}, train acc: {training_acc / num_batch: >8.4f}")
        logger.info(f"==========>>Epoch {epoch} done<<==========")
        pass
    pass
    return model


def evaluate(model, samples_dataloader):
    pass


def app():
    # 训练超参数
    batch_size = 64
    num_epoch = 5

    # 加载原始数据集
    train_dataset, test_dataset = AG_NEWS(root="/home/souche/data/torch/", ngrams=3)
    # 装载数字化数据样本
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)  # 训练集洗牌
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # 输入参数
    vocab_size = len(train_dataset.get_vocab())
    embed_dim = 128
    num_class = 4

    # 实例化训练模型
    tc_model = TextClassificationModel(vocab_size, embed_dim, num_class)
    model = training(tc_model, train_dataloader, num_epoch)
    print()
    pass


if __name__ == '__main__':
    app()
