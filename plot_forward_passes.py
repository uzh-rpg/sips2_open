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

import cProfile

import sips2.evaluate as evaluate
import sips2.flags as flags  # Needs to be imported to parse.
import sips2.hyperparams as hyperparams
import sips2.system as system

FLAGS = flags.FLAGS


def run():
    hyperparams.announceEval()
    eval_pairs = hyperparams.getEvalDataGen()
    graph, sess = hyperparams.modelFromCheckpoint()
    forward_passer = hyperparams.getForwardPasser(graph, sess)
    fp_cache = system.ForwardPassCache(forward_passer)
    for pair_i in range(len(eval_pairs)):
        print('%d/%d' % (pair_i, len(eval_pairs)))
        pair = eval_pairs[pair_i]
        print(pair.name())
        fps = [fp_cache[im] for im in pair.im]

        assert len(fps[0]) == FLAGS.num_test_pts
        assert len(fps[1]) == FLAGS.num_test_pts
        matched_indices = system.match(fps)
        inliers = system.getInliers(pair, fps, matched_indices)
        matched_fps = [fps[i][matched_indices[i]] for i in [0, 1]]
        evaluate.renderMatching(pair, fps, matched_fps, inliers)


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    pr.dump_stats('plot_forward_passes.profile')
