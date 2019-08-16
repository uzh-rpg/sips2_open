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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sips2.flags  # required to parse flags
import sips2.hyperparams as hyperparams

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if __name__ == '__main__':
    score_wasinlier = np.loadtxt(hyperparams.wasInlierFilePath())
    num_bins = 20
    bins = [[] for _ in range(num_bins)]
    for row in range(score_wasinlier.shape[0]):
        index = int(score_wasinlier[row, 0] * num_bins)
        bins[index].append(score_wasinlier[row, 1])

    frequencies = np.array([np.mean(bin_) for bin_ in bins])

    counts = np.array([len(bin_) for bin_ in bins])

    bin_width = float(1) / num_bins

    fig, ax1 = plt.subplots(figsize=[5, 2.5])
    ax1.hlines(frequencies, bin_width * np.arange(num_bins),
               bin_width * (np.arange(num_bins) + 1), colors='red', linewidth=3)
    ax1.plot([0, 1], [0, 1], color='black')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Predicted inlierness probability')
    ax1.set_ylabel('Inlierness frequency', color='r')
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_ylabel('# points with given prediction', color='b')
    ax2.stem((bin_width * (np.arange(num_bins) + 0.5))[counts > 0],
             counts[counts > 0], color='b',
             basefmt='')
    ax2.set_ylim(bottom=0)

    # ax1.set_title('Probability prediction on %s' % hyperparams.evalString(k=False))

    plt.tight_layout()

    plt.savefig('%s.pdf' % hyperparams.wasInlierFilePath())
    plt.show()
