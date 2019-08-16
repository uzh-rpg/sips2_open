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

import absl.flags
import matplotlib.pyplot as plt

import sips2.hyperparams as hyperparams
import plot_pose_accuracy

FLAGS = absl.flags.FLAGS


def plot(color=plt.get_cmap('tab10').colors[0]):
    ks = [5, 6, 7, 8, 10, 12, 15, 20, 25, 30]
    original_k = FLAGS.k
    aucs = []
    for k in ks:
        FLAGS.k = k
        aucs.append(plot_pose_accuracy.plot(do_plot=False))
    FLAGS.k = original_k
    saucs = [i[0] for i in aucs]
    raucs = [i[1] for i in aucs]
    taucs = [i[2] for i in aucs]

    plt.plot(ks, saucs, label=hyperparams.label(), color=color)
    plt.plot(ks, raucs, '--', color=color)
    plt.plot(ks, taucs, ':', color=color)


def limsAndLabels():
    plt.xlim([4, 30])
    plt.xlabel('k')
    plt.ylim([0, 1])
    plt.ylabel('AUC-1')
    plt.grid()
    plt.legend()


if __name__ == '__main__':
    plt.figure(0, figsize=[5, 3])

    if FLAGS.baseline == 'all':
        cmap = plt.get_cmap('tab10')
        for bl_ci in zip(['', 'sift', 'surf', 'super'], range(4)):
            FLAGS.baseline = bl_ci[0]
            plot(color=cmap.colors[bl_ci[1]])
        FLAGS.baseline = 'all'
    else:
        plot()
    limsAndLabels()
    # plt.title('influence of k, %s' % hyperparams.methodEvalString(k=False))
    plt.tight_layout()
    if FLAGS.baseline == 'all':
        plt.savefig('plots/varyk_%s.pdf' % hyperparams.evalString())
    else:
        plt.show()
