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

from IPython.core import ultratb
import matplotlib.pyplot as plt
import numpy as np
import sys

import sips2.baselines as baselines
import sips2.cache_forward_pass as cache_forward_pass
import sips2.evaluate as evaluate
import sips2.flags as flags
import sips2.hyperparams as hyperparams
import sips2.multiscale as multiscale
import sips2.system as system

FLAGS = flags.FLAGS


def plot(label=None, k=10):
    hyperparams.announceEval()
    eval_pairs = hyperparams.getEvalDataGen()
    # Very special case lfnet:
    if FLAGS.baseline == 'lfnet':
        x = [50, 100, 150]
        y = []
        for num_pts in x:
            forward_pass_dict = baselines.parseLFNetOuts(eval_pairs, num_pts)
            success = np.zeros(len(eval_pairs), dtype=bool)
            for pair_i in range(len(eval_pairs)):
                pair = eval_pairs[pair_i]
                folder, a, b = pair.name().split(' ')
                forward_passes = [forward_pass_dict['%s%s' % (folder, i)]
                                  for i in [a, b]]

                matched_indices = system.match(forward_passes)
                inliers = system.getInliers(
                    pair, forward_passes, matched_indices)
                if np.count_nonzero(inliers) >= k:
                    success[pair_i] = True
            y.append(np.mean(success.astype(float)))
        plt.plot(x, y, 'x', label='%s: N/A' % (hyperparams.methodString()))
    else:
        pair_outs = cache_forward_pass.loadOrCalculateOuts()
        if FLAGS.num_scales > 1 and FLAGS.baseline == '':
            fps = [[multiscale.forwardPassFromHicklable(im) for im in pair]
                   for pair in pair_outs]
        else:
            fps = [[system.forwardPassFromHicklable(im) for im in pair]
                   for pair in pair_outs]
        pairs_fps = zip(eval_pairs, fps)
        stats = [evaluate.leastNumForKInliers(pair_fps[0], pair_fps[1], k)
                 for pair_fps in pairs_fps]
        x, y = evaluate.lessThanCurve(stats)
        auc = evaluate.auc(x, y, 200)

        plt.step(x, y, label='%s: %.2f' % (hyperparams.label(), auc))
        return auc


def limsAndLabels():
    plt.xlim([0, 200])
    plt.xlabel('Extracted interest point count')
    plt.ylim([0, 1])
    plt.ylabel('Success rate')
    plt.grid()
    plt.legend()


if __name__ == '__main__':
    sys.excepthook = ultratb.FormattedTB(
        color_scheme='Linux', call_pdb=1)

    plt.figure(0, figsize=[5, 2.5])

    if FLAGS.baseline == 'all':
        for bl in ['', 'sift', 'surf', 'super', 'lfnet']:
            FLAGS.baseline = bl
            plot()
        FLAGS.baseline = 'all'
    else:
        plot()
    limsAndLabels()
    # plt.title('Succinctness on %s' % hyperparams.evalString())
    plt.tight_layout()
    if FLAGS.baseline == 'all':
        plt.savefig('plots/succ_%s.pdf' % hyperparams.evalString())
    else:
        plt.show()
