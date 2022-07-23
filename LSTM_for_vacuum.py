# import library

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable

training_set = pd.read_csv()


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length -1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)
