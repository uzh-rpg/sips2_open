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
import os

import baselines
import evaluate
import flags  # Needs to be imported to parse.
import hyperparams
import sequences
import system

FLAGS = flags.FLAGS


def loadOrEvaluate():
    """ Returns [n], [R_err], [t_err]. """
    if not os.path.exists(hyperparams.evalPath()):
        evaluateAndSave()
    elif os.path.getmtime(hyperparams.checkpointPath() + '_stats') > \
        os.path.getmtime(hyperparams.evalPath()) \
            and FLAGS.baseline == '':
        evaluateAndSave()
    result = hkl.load(open(hyperparams.evalPath(), 'r'))
    print('Evaluated on %d pairs' % len(result[0]))
    return result


def evaluateAndSave():
    print('RUNNING EVALUATION...')
    eval_pairs = hyperparams.getEvalDataGen()
    pair_outs = loadOrCalculateOuts()
    get_rt = type(eval_pairs[0]) == sequences.PairWithStereo

    fps = [[system.forwardPassFromHicklable(im) for im in pair]
           for pair in pair_outs]
    pairs_fps = zip(eval_pairs, fps)
    stat_Rerr_terr = [evaluate.leastNumForKInliers(
        pair_fps[0], pair_fps[1], FLAGS.k, get_rt=get_rt)
        for pair_fps in pairs_fps]
    if get_rt:
        result = [[i[j] for i in stat_Rerr_terr] for j in range(3)]
    else:
        result = [stat_Rerr_terr, None, None]
    hkl.dump(result, open(hyperparams.evalPath(), 'w'))


def loadOrCalculateOuts():
    if not os.path.exists(hyperparams.cachedForwardPath()):
        cache()
    elif os.path.getmtime(hyperparams.checkpointPath() + '_stats') > \
        os.path.getmtime(hyperparams.cachedForwardPath()) \
            and FLAGS.baseline == '':
        cache()
    return hkl.load(open(hyperparams.cachedForwardPath(), 'r'))


def cache():
    hyperparams.announceEval()
    eval_pairs = hyperparams.getEvalDataGen()
    pair_outs = []
    if FLAGS.baseline == 'super':
        forward_pass_dict = baselines.parseSuperPointOuts(eval_pairs)
        for pair_i in range(len(eval_pairs)):
            pair = eval_pairs[pair_i]
            folder, a, b = pair.name().split(' ')
            forward_passes = [forward_pass_dict['%s%s' % (folder, i)]
                              for i in [a, b]]
            pair_outs.append([fp.hicklable() for fp in forward_passes])
    else:
        graph, sess = hyperparams.modelFromCheckpoint()
        forward_passer = hyperparams.getForwardPasser(graph, sess)
        fp_cache = system.ForwardPassCache(forward_passer)
        for pair_i in range(len(eval_pairs)):
            print('%d/%d' % (pair_i, len(eval_pairs)))
            pair = eval_pairs[pair_i]
            print(pair.name())
            fps = [fp_cache[im] for im in pair.im]
            pair_outs.append([fp.hicklable() for fp in fps])

    hkl.dump(pair_outs, open(hyperparams.cachedForwardPath(), 'w'))


if __name__ == '__main__':
    cache()
