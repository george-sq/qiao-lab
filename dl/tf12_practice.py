#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date   : 2021/4/13 15:41
@Author : qiangzi
@File   : tf12_practice.py
@Todo   : TensorFlow1.12练习
"""

import logging

import tensorflow as tf

__log_format = "%(asctime)s [ %(levelname)-7s ] | %(filename)-24s(line:%(lineno)-4s)-%(process)d(%(thread)d) || %(message)s"
__date_format = "%Y-%m-%d(%A) %H:%M:%S(%Z)"
logging.basicConfig(level=logging.DEBUG, format=__log_format, datefmt=__date_format)
logger = logging.getLogger(__name__)


def hello_tf():
    hello = tf.constant("Hello Tensorflow")
    sess = tf.Session()
    print(sess.run(hello))

    pass


def gpu_test():
    from tensorflow.python.client import device_lib

    # print(tf.test.is_gpu_available())
    print(f"GPU test: {tf.compat.v1.test.is_gpu_available()}")
    # 列出所有的本地机器设备
    local_device_protos = device_lib.list_local_devices()
    print()
    print(f"local devices: {local_device_protos}")
    print()

    # 只打印GPU设备
    l_gpu_devices = [x for x in local_device_protos if 'GPU' in x.device_type]
    print()
    print(f"local GPU devices: {l_gpu_devices}")
    print()
    pass


def app():
    # hello tensorflow
    # hello_tf()

    # gpu_test
    # gpu_test()

    pass


if __name__ == '__main__':
    app()
