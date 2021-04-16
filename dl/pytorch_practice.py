#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date   : 2021/3/29 10:09
@Author : qiangzi
@File   : pytorch_practice.py
@Todo   : something
"""

import logging

import torch
from torch import nn
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

__all__ = []

__log_format = "%(asctime)s [ %(levelname)-7s ] | %(filename)-24s(line:%(lineno)-4s)-%(process)d(%(thread)d) || %(message)s"
__date_format = "%Y-%m-%d(%A) %H:%M:%S(%Z)"
logging.basicConfig(level=logging.DEBUG, format=__log_format, datefmt=__date_format)
logger = logging.getLogger(__name__)


class NeuralNetwork(nn.Module):
    """
    Define Neural Network model
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Module.Flatten()
        self.flatten = Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class BasicDeepLearning(object):

    _allowed_devices = ("cuda", "cpu")

    def __init__(self, model=None,
                 train_data=None, valid_data=None, test_data=None,
                 lr=1e-3, momentum=0,
                 epochs=1, batch_size=64, device=None):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        if device in self._allowed_devices:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pass

    def _setting(self):
        # 设置网络
        assert isinstance(self.model, nn.Module)
        # 设置损失评估
        loss_fn = nn.CrossEntropyLoss()
        # 设置优化方案
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        pass
        return loss_fn, optimizer

    def training(self):
        assert isinstance(self.train_data, DataLoader)
        assert isinstance(self.epochs, int) and self.epochs > 0
        loss_fn, optimizer = self._setting()
        self.model.to(self.device)
        self.model.train()  # 开启模型的训练模式
        for epoch in range(self.epochs):
            print(f"[Epoch {epoch}] starting ...")
            batch_totals = len(self.train_data)
            # 训练
            for batch, (X, y) in enumerate(self.train_data):
                X, y = X.to(self.device), y.to(self.device)
                # 前向计算
                y_out = self.model(X)
                # 计算误差
                loss = loss_fn(y_out, y)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                # 参数优化
                optimizer.step()
                # 进度输出
                if batch % 100 == 0:
                    loss, trained_samples = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}, batch:{batch:>5d} /{batch_totals:>5d}, trained samples:{trained_samples:>7d}")
                pass
            # 交叉验证
            # assert isinstance(self.valid_data, DataLoader)
            print(f"[Epoch {epoch}] done")
            print()
            pass
        pass

    def evaluation(self):
        assert isinstance(self.model, nn.Module)
        assert isinstance(self.test_data, DataLoader)
        loss_fn, _ = self._setting()
        self.model.eval()  # 开启模型的评估模式，会固定相关参数
        loss, correct_totals = 0, 0
        with torch.no_grad():
            batch_totals = len(self.test_data)
            for X, y in self.test_data:
                X, y = X.to(self.device), y.to(self.device)
                y_out = self.model(X)
                loss += loss_fn(y_out, y).item()  # batch size loss累加
                # pred_results = (y_pred.argmax(1) == y)
                # correct += pred_results.type(torch.float).sum().item()
                correct_totals += (y_out.argmax(1) == y).type(torch.float).sum().item()
                pass
            pass
        loss /= batch_totals
        acc = correct_totals / (batch_totals * self.batch_size)
        print(f"Test Error: \n Accuracy: {(100 * acc):>0.1f}%, Avg loss: {loss:>8f} \n")
        pass

    def save(self, path=None):
        pass

    def load(self, path=None):
        pass
    pass


def quickstart():
    # 数据集加载
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    print(f"training dataset: {len(training_data)}")
    print(f"test dataset: {len(test_data)}")
    print()

    # 数据样本装载
    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    # 样本查看
    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # 实例化神经网络模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)
    print(model)

    # 定义训练过程
    dl = BasicDeepLearning(model=model, train_data=train_dataloader, test_data=test_dataloader, batch_size=train_dataloader.batch_size, epochs=5)
    # 模型训练
    dl.training()
    # 模型评估
    dl.evaluation()
    print()
    pass


def app():
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # quickstart
    quickstart()
    pass


if __name__ == '__main__':
    app()
