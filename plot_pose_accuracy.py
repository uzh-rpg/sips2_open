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

import sips2.cache_forward_pass as cache_forward_pass
import sips2.evaluate as evaluate
import sips2.flags as flags
import sips2.hyperparams as hyperparams

FLAGS = flags.FLAGS


def plot(do_plot=True):
    hyperparams.announceEval()
    succ, Rerr, terr = cache_forward_pass.loadOrEvaluate()
    assert Rerr is not None
    sx, sy = evaluate.lessThanCurve(succ)
    sauc = evaluate.auc(sx, sy, 200)
    rx, ry = evaluate.lessThanCurve(Rerr)
    if FLAGS.ds == 'eu':
        print('5 degrees')
        rmax = 5
    else:
        rmax = 1
    rauc = evaluate.auc(rx, ry, rmax)
    tx, ty = evaluate.lessThanCurve(terr)
    tauc = evaluate.auc(tx, ty, 1)

    if do_plot:
        plt.step(
            rx, ry, label='%s R: %.2f' % (hyperparams.methodString(), rauc))
        plt.step(
            tx, ty, label='%s t: %.2f' % (hyperparams.methodString(), tauc))
    return sauc, rauc, tauc


def limsAndLabels():
    if FLAGS.ds == 'eu':
        print('5 degrees')
        rmax = 5
    else:
        rmax = 1
    plt.xlim([0, rmax])
    plt.xlabel('Error [m/degrees]')
    plt.ylim([0, 1])
    plt.ylabel('Fraction with lower error')
    plt.grid()
    plt.legend()


if __name__ == '__main__':
    if FLAGS.baseline == 'all':
        for bl in ['', 'sift', 'surf']:
            FLAGS.baseline = bl
            plot()
        FLAGS.baseline = 'all'
    else:
        plot()
    limsAndLabels()
    plt.title('%s' % hyperparams.evalString())
    plt.show()
