# Copyright (C) 2019 Titus Cieslewski, RPG, University of Zurich, Switzerland
#   You can contact the author at <titus at ifi dot uzh dot ch>
# Copyright (C) 2019 Konstantinos G. Derpanis,
#   Dept. of Computer Science, Ryerson University, Toronto, Canada
# Copyright (C) 2019 Davide Scaramuzza, RPG, University of Zurich, Switzerland
#
# This file is part of sips2_open.
#
# sips2_open is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# sips2_open is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with sips2_open. If not, see <http:#www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np


def inlLoss(x):
    return -np.log(x)


def outlLoss(x):
    return -np.log(1-x)


def dInlLoss(x):
    return 1/x


def dOutlLoss(x):
    return -1/(1-x)


if __name__ == '__main__':
    plt.figure(figsize=[6.4, 2.4])

    eps = 1e-3
    x = np.linspace(eps, 1-eps, 100)
    inly = inlLoss(x)
    outly = outlLoss(x)
    plt.plot(x, inly, 'g')
    plt.plot(x, outly, 'r')

    x_sample = 0.7
    plt.axvline(x=x_sample, color='black')
    plt.arrow(x_sample, inlLoss(x_sample),
              dInlLoss(x_sample)/10, 0, color='g', width=0.01)
    plt.arrow(x_sample, outlLoss(x_sample),
              dOutlLoss(x_sample) / 10, 0, color='r', width=0.01)

    plt.ylim([0, 2])
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('predicted probability')
    plt.tight_layout()
    plt.show()
