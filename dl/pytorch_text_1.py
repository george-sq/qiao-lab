#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date   : 2021/3/29 17:35
@Author : qiangzi
@File   : pytorch_text_1.py
@Todo   : pytorch_text_1
"""

import logging
from io import open
import glob
import os
import unicodedata
import string
import time
import math
import random
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

import torch
import torch.nn as nn

__all__ = []

__log_format = "%(asctime)s [ %(levelname)-7s ] | %(filename)-24s(line:%(lineno)-4s)-%(process)d(%(thread)d) || %(message)s"
__date_format = "%Y-%m-%d(%A) %H:%M:%S(%Z)"
logging.basicConfig(level=logging.INFO, format=__log_format, datefmt=__date_format)
logger = logging.getLogger(__name__)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

path_txt = '../resource/data/names/*.txt'


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s):
    """
    从 Unicode 编码转换为 ASCII 编码
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Read a file and split into lines
def readLines(path_filename):
    l_lines = open(path_filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in l_lines]


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, vec_input, vec_hidden):
        combined = torch.cat((vec_input, vec_hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def get_raw_samples():
    l_files = findFiles(path_txt)
    # print(l_files)

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    for filename in l_files:
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    return all_categories, category_lines


def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(all_categories, category_lines):
    one_category = randomChoice(all_categories)
    sample_line = randomChoice(category_lines[one_category])
    category_tensor = torch.tensor([all_categories.index(one_category)], dtype=torch.long)
    line_tensor = lineToTensor(sample_line)
    return one_category, sample_line, category_tensor, line_tensor


def train(model, hidden, n_hidden, n_categories, category_tensor, line_tensor):
    learning_rate = 0.005
    # rnn = RNN(n_letters, n_hidden, n_categories)
    criterion = nn.NLLLoss()

    # hidden = model.initHidden()
    model.zero_grad()
    output = None
    size_line_tensor = line_tensor.size()
    print(size_line_tensor)
    print(f"----------------样本训练--------------")
    for i in range(size_line_tensor[0]):
        print(f"input index: {i}")
        output, hidden = model(line_tensor[i], hidden)
        print(f"hidden: {hidden}, size: {hidden.size()}")
        print(f"output: {output}, size: {output.size()}")
        print()
        # output.topk(3)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    print(f"===============参数更新===============")
    for p in model.parameters():
        print(f"before: {p}, size: {p.size()}")
        p.data.add_(p.grad.data, alpha=-learning_rate)
        print(f"after: {p}, size: {p.size()}")
        print()

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def app():
    print(unicodeToAscii('Ślusàrski'))
    print()
    # 加载原始数据
    all_categories, category_lines = get_raw_samples()
    n_categories = len(all_categories)
    print(f"category size: {n_categories}, categories: {all_categories}")
    # print(letterToTensor('J'))
    # print(lineToTensor('Jones').size())
    print()

    n_hidden = 128
    # rnn = RNN(n_letters, n_hidden, n_categories)

    # ts_input = letterToTensor('A')
    # ts_hidden = torch.zeros(1, n_hidden)

    # ts_output, ts_next_hidden = rnn(ts_input, ts_hidden)
    # print(categoryFromOutput(ts_output, all_categories))

    # for i in range(10):
    #     category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
    #     print('category =', category, '/ line =', line)
    # print()

    # 训练loop示例
    n_iters = 10000
    print_every = 500
    plot_every = 100

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    rnn = RNN(n_letters, n_hidden, n_categories)
    hidden = rnn.initHidden()
    for i in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories, category_lines)
        output, loss = train(rnn, hidden, n_hidden, n_categories, category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if i % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if i % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.show()
    print()
    pass


if __name__ == '__main__':
    app()
