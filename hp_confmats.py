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

import hickle as hkl
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as scidist

import sips2.evaluate as evaluate
import sips2.flags as flags
import sips2.hyperparams as hyperparams
import sips2.multiscale as multiscale
import sips2.system as system

FLAGS = flags.FLAGS


if __name__ == '__main__':
    hyperparams.announceEval()
    assert FLAGS.ds == 'hp'
    eval_pairs = hyperparams.getEvalDataGen()
    pair_outs = hkl.load(open(hyperparams.cachedForwardPath(), 'r'))
    if FLAGS.num_scales > 1 and FLAGS.baseline == '':
        unpacker = multiscale.forwardPassFromHicklable
    else:
        unpacker = system.forwardPassFromHicklable
    fps = [[unpacker(im) for im in pair] for pair in pair_outs]
    pairs_fps = zip(eval_pairs, fps)
    stats = [evaluate.leastNumForKInliers(pair_fps[0], pair_fps[1], 10)
             for pair_fps in pairs_fps]

    n = len(eval_pairs.folder_names)
    assert len(stats) == 15 * n

    w = 3
    h = int(np.ceil(float(n) / w))
    im = np.zeros([8 * h, 8 * w])
    title = hyperparams.methodEvalString() + ':'
    for i in range(n):
        row = i / w
        col = i % w
        squ = scidist.squareform(stats[i * 15:(i+1) * 15])
        im[(row * 8 + 1):(row * 8 + 7), (col * 8 + 1):(col * 8 + 7)] = squ
        if col == 0:
            title = title + '\n'
        title = title + eval_pairs.folder_names[i] + ', '

    plt.imshow(im)
    plt.colorbar()
    plt.title(title)
    plt.show()
