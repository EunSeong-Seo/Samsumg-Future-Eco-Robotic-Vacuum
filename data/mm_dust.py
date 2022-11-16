import numpy as np
import torch
import torch.utils.data as data

import os
import random
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

random.seed(1996)


class DustType(Enum):
    Sine = 0  # Sinusoidal
    SI = 1  # Slowly Increase
    FI = 2  # Fastly Increase


class DustMAP(data.Dataset):
    def __init__(self, is_train=True, n_frames_input=10, n_frames_output=10):
        super(DustMAP, self).__init__()

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.image_size_ = 64
        self.data = None

        # the characteristic of dust map
        self.min_value = 0.
        self.max_value = 5.

        # the location of Test Data
        self.data_dir = Path("./Dust_Dataset")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # the location of Figure
        self.fig_dir = Path("./Figure")
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.filename = "Dust_Sequence.png"

        # Split Sector
        # ----------------------------------------------------------------------
        # slowly increase
        self.si = [(i, j) for i in range(3, 30) for j in range(23, 43)]
        self.si.extend([(i, j) for i in range(40, 50) for j in range(40, self.image_size_)])

        # Fastly increase
        self.fi = [(i, j) for i in range(0, 40) for j in range(0, 10)]
        self.fi.extend([(i, j) for i in range(0, 20) for j in range(54, self.image_size_)])
        self.fi.extend([(i, j) for i in range(40, 64) for j in range(0, 7)])
        self.fi.extend([(i, j) for i in range(50, 64) for j in range(7, 25)])
        self.fi.extend([(i, j) for i in range(30, 35) for j in range(20, 46)])

        # Remain part will be automatically sinusoidal part
        # ----------------------------------------------------------------------
        self.draw_sector()
        # self.save_data()

    def __getitem__(self, idx):

        # Make data
        self.data = self.make_data()

        # Add Noise
        # self.data = np.clip(self.data + self.gaussian_noise(self.data.shape), self.min_value, self.max_value)
        self.data = np.clip(self.data, self.min_value, self.max_value)
        input = self.data[:self.n_frames_input]
        output = self.data[self.n_frames_input:self.n_frames_total]
        frozen = input[-1]

        output = torch.from_numpy(output / self.max_value).float()
        input = torch.from_numpy(input / self.max_value).float()

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.n_frames_total

    def draw_sector(self):
        sector = np.zeros((self.image_size_, self.image_size_))
        for i in self.si:
            sector[i] = DustType.SI.value
        for i in self.fi:
            sector[i] = DustType.FI.value

        fig = plt.figure()
        fig.set_size_inches(15, 15)
        sns.heatmap(sector, annot=True)
        plt.title('Dust Map Sector', fontsize=20)
        fig.savefig("Figure/sector.png")
        plt.cla()

    def plot_sequence(self, data, filename):
        assert data.shape == (self.n_frames_total, 1)
        print(np.arange(1, self.n_frames_total + 1,
                        step=1, dtype=int).shape)
        print(data[:, 0].shape)
        df = pd.DataFrame({"Time Step": np.arange(1, self.n_frames_total + 1,
                                                  step=1, dtype=int), "Dust Amount": data[:, 0]})
        sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor': 'black'})
        sns_plot = sns.lineplot(x="Time Step", y="Dust Amount", data=df)
        fig = sns_plot.get_figure()
        fig.savefig(self.fig_dir.joinpath(filename))
        plt.cla()

    def make_data(self):
        data = np.empty((self.n_frames_total, 1, self.image_size_, self.image_size_))  # (sequence, c, h, w)

        for h in range(self.image_size_):
            for w in range(self.image_size_):
                if (h, w) in self.si:
                    # slow increase sector
                    start = random.random() * 3  # min : 0 max : 3
                    end = start + random.random()  # min : start, max : start + 1
                    data[:, 0, h, w] = self.linear_func(start, end, self.n_frames_total)
                elif (h, w) in self.fi:
                    # fast increase sector
                    start = random.random() * 2  # min : 0 max : 2
                    end = start + random.random() * 3  # min : start, max : start + 3
                    data[:, 0, h, w] = self.linear_func(start, end, self.n_frames_total)
                else:
                    # sinusoidal sector
                    mean = 1.5 + random.random() * 1.5  # min : 1.5 max : 3
                    amplitude = random.random() * 1.5  # min : 0 max : 1.5
                    data[:, 0, h, w] = self.sin_func(mean, amplitude, self.n_frames_total)

        return data

    def save_data(self, data):
        # np.save(self.dir / "train", self.data[:int(self.num_data * self.train_ratio), :, :])
        # np.save(self.dir / "valid", self.data[int(self.num_data * self.train_ratio): int(self.num_data * (1-self.test_ratio)), :, :])
        np.save(self.data_dir / "test", data)

    def sin_func(self, mean, amplitude, num_data):
        start = random.random() * 2 * math.pi
        end = start + random.random() * math.pi * 10
        theta = np.linspace(start, end, num_data)
        data = mean + amplitude * np.sin(theta)
        return data

    def linear_func(self, start, stop, num_data):
        data = np.linspace(start, stop, num_data)
        return data

    def gaussian_noise(self, size):
        return np.random.normal(0., 0.01, size=size)


if __name__ == "__main__":
    generator = DustMAP()
    generator.__getitem__(0)