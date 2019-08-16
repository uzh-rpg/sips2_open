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

import numpy as np

import sips2.flags as flags
import sips2.hyperparams as hyperparams
import sips2.system as system

FLAGS = flags.FLAGS


if __name__ == '__main__':
    hyperparams.announceEval()
    eval_pairs = hyperparams.getEvalDataGen()
    graph, sess = hyperparams.modelFromCheckpoint()
    forward_passer = hyperparams.getForwardPasser(graph, sess)
    score_wasinlier = []
    for pair in eval_pairs:
        fps = [forward_passer(im) for im in pair.im]
        matched_indices = system.match(fps)
        inlier_mask = system.getInliers(pair, fps, matched_indices)
        for im_i in [0, 1]:
            im = pair.im[im_i]
            inlier_indices = matched_indices[im_i][inlier_mask]
            scores = fps[im_i].ip_scores
            for pt_i in range(len(scores)):
                score_wasinlier.append([scores[pt_i], pt_i in inlier_indices])

    np.savetxt(hyperparams.wasInlierFilePath(), np.array(score_wasinlier))