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
import sips2.system as system

FLAGS = flags.FLAGS


YMAX = 100


def plot(color=plt.get_cmap('tab10').colors[0]):
    n = FLAGS.num_test_pts
    hyperparams.announceEval()
    eval_pairs = hyperparams.getEvalDataGen()
    # Very special case lfnet:
    if FLAGS.baseline == 'lfnet':

        forward_pass_dict = baselines.parseLFNetOuts(eval_pairs, n)
        fps = []
        for pair_i in range(len(eval_pairs)):
            pair = eval_pairs[pair_i]
            folder, a, b = pair.name().split(' ')
            forward_passes = [forward_pass_dict['%s%s' % (folder, i)]
                              for i in [a, b]]
            matched_indices = system.match(forward_passes)
            fps.append([forward_passes[i][matched_indices[i]] for i in [0, 1]])
    else:
        pair_outs = cache_forward_pass.loadOrCalculateOuts()
        if FLAGS.num_scales > 1 and FLAGS.baseline == '':
            raise NotImplementedError
        else:
            fps = []
            full_fps = []
            for pair in pair_outs:
                reduced = [system.forwardPassFromHicklable(i).reducedTo(n)
                           for i in pair]
                full_fps.append(reduced)
                matched_indices = system.match(reduced)
                fps.append([reduced[i][matched_indices[i]] for i in [0, 1]])

    pairs_fps = zip(eval_pairs, fps)
    masks_errs = [evaluate.p3pMaskAndError(pair_fps[0], pair_fps[1])
                  for pair_fps in pairs_fps]

    if FLAGS.baseline == '':
        for mask_e, pair_fps, full in zip(masks_errs, pairs_fps, full_fps):
            mask, rerr, terr = mask_e
            pair, fps = pair_fps
            evaluate.renderMatching(pair, full, fps, mask)

    ninl = np.array([np.count_nonzero(i[0]) for i in masks_errs])
    rerrs = np.array([i[1] for i in masks_errs])
    rerrs[ninl < 10] = np.inf
    terrs = np.array([i[2] for i in masks_errs])
    terrs[ninl < 10] = np.inf

    if FLAGS.baseline != 'sift':
        rlabel, tlabel = hyperparams.label(), None
    else:
        rlabel, tlabel = '%s, rot.' % hyperparams.label(), \
                         '%s, transl.' % hyperparams.label()

    x, y = evaluate.lessThanCurve(rerrs)
    plt.semilogx(x, y, '--', color=color, label=rlabel)
    x, y = evaluate.lessThanCurve(terrs)
    plt.semilogx(x, y, ':', color=color, label=tlabel)



def limsAndLabels():
    plt.xlim([0.01, 10])
    plt.xlabel('rotation, translation errors[deg], [m]')
    plt.ylim([0, 1])
    plt.ylabel('fraction with lower error')
    plt.grid()
    plt.legend()


if __name__ == '__main__':
    sys.excepthook = ultratb.FormattedTB(
        color_scheme='Linux', call_pdb=1)

    plt.figure(0, figsize=[5, 3])
    cmap = plt.get_cmap('tab10')
    if FLAGS.baseline == 'all':
        for bl, color in zip(
                ['', 'sift', 'surf', 'super', 'lfnet'], cmap.colors):
            FLAGS.baseline = bl
            plot(color=color)
        FLAGS.baseline = 'all'
    else:
        plot()
    limsAndLabels()
    plt.tight_layout()
    if FLAGS.baseline == 'all':
        plt.savefig('plots/errs_%s.pdf' % hyperparams.evalString(k=False))
    else:
        plt.show()
