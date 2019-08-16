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

import csv
import IPython
import hyperparams
import math
import matplotlib.pyplot as plt
import numpy as np
import os

import rpg_common_py.geometry

import sips2.cache_forward_pass as cache_forward_pass
import sips2.evaluate as evaluate
import sips2.flags as flags
import sips2.multiscale as multiscale
import sips2.system as system

FLAGS = flags.FLAGS


if __name__ == '__main__':
    hyperparams.announceEval()

    eval_pairs = hyperparams.getEvalDataGen()

    assert FLAGS.baseline == ''
    graph, sess = hyperparams.modelFromCheckpoint()
    forward_passer = hyperparams.getForwardPasser(graph, sess)

    pair_outs = cache_forward_pass.loadOrCalculateOuts()
    if FLAGS.num_scales > 1 and FLAGS.baseline == '':
        fps = [[multiscale.forwardPassFromHicklable(im) for im in pair]
               for pair in pair_outs]
    else:
        fps = [[system.forwardPassFromHicklable(im) for im in pair]
               for pair in pair_outs]
    pairs_fps = zip(eval_pairs, fps)
    stats = [evaluate.leastNumForKInliers(
        pair_fps[0], pair_fps[1], 10, get_rt=True) for pair_fps in pairs_fps]

    nmins = [i[0] for i in stats]
    Rerrs = [i[1] for i in stats]
    terrs = [i[2] for i in stats]

    # Also save diffs of evaluation set
    dR, dt = [], []
    for pair in eval_pairs:
        T = pair.T_0_1
        dR.append(math.degrees(rpg_common_py.geometry.getRotationAngle(T.R)))
        dt.append(np.linalg.norm(T.t))

    # Save csv
    csv_dir = os.path.join('results', 'match_render',
                           hyperparams.methodEvalString())
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, '%s.csv' % hyperparams.methodEvalString())
    csv_vals = zip([i.name() for i in eval_pairs], dR, dt, nmins, Rerrs, terrs)
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=' ', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['name', 'dR', 'dt', 'nmin', 'eR', 'et'])
        for val in csv_vals:
            writer.writerow(val)

    dR = np.array(dR)
    dt = np.array(dt)
    nmins = np.array(nmins)
    mt100filt = nmins > 100
    lt50filt = nmins < 50
    btfilt = np.logical_not(mt100filt) & np.logical_not(lt50filt)

    plt.figure(0, figsize=[5, 3])
    plt.plot(dR[btfilt], dt[btfilt], 'o', label='50-100 required')
    plt.plot(dR[mt100filt], dt[mt100filt], '^', label='>100 required')
    plt.plot(dR[lt50filt], dt[lt50filt], 'v', label='< 50 required')
    plt.grid()
    plt.legend()
    plt.xlabel('angle difference [degrees]')
    plt.ylabel('distance [meters]')
    # plt.title('Dataset characterization %s' % hyperparams.evalString())
    plt.tight_layout()
    plt.savefig('plots/drdt_%s.pdf' % hyperparams.evalString(k=False))
