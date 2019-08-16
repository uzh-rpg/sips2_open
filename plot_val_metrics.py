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

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import os

import sips2.flags as flags
import sips2.hyperparams as hyperparams

FLAGS = flags.FLAGS


def doPlot(_):
    plt.clf()
    path = hyperparams.trainStatsPath()
    if os.path.exists(path):
        stats = np.loadtxt(path)
        if len(stats.shape) < 2:
            stats = stats.reshape([1, stats.size])

        plt.plot(stats[:, 0], stats[:, 1], label='Succ. AUC-200 Val.')
        if stats.shape[1] > 2:
            plt.plot(stats[:, 0], stats[:, 2], label='Succ. AUC-200 Train')
        if stats.shape[1] > 3:
            raise Exception('Deprecated')
            plt.plot(stats[:, 0], stats[:, 2], label='R_err AUC-1')
            plt.plot(stats[:, 0], stats[:, 3], label='t_err AUC-1')
        plt.grid()
        plt.title(hyperparams.shortString())
        plt.ylabel('AUC')
        plt.ylim(bottom=0)
        plt.xlabel('training iterations')
        plt.legend()
    else:
        plt.title('Checkpoint not yet around.')


if __name__ == '__main__':
    hyperparams.announceEval()
    fig = plt.figure()

    ani = matplotlib.animation.FuncAnimation(
        fig, doPlot, repeat=True, interval=1000)

    plt.show()
