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
import hickle as hkl
import numpy as np
import os

import sips2.evaluate as evaluate
import sips2.hyperparams as hyperparams
import sips2.p3p as p3p
import sips2.sequences as sequences
import sips2.system as system


def run():
    seqs = hyperparams.getEvalSequences()
    for seq in seqs:
        wrapper = sequences.Wrapper(seq)
        print(seq.name())
        spath = os.path.join(hyperparams.seqFpsPath(), seq.name())
        assert os.path.exists(spath)
        n = 0
        while os.path.exists(os.path.join(spath, '%05d.jpg' % n)):
            n = n + 1

        confmat = np.ones((n, n), dtype=int) * 500
        least_nums = []
        stereo_cache = p3p.StereoCache(n)
        for i in range(n):
            print('%d/%d' % (i, n))
            fp_i = system.forwardPassFromHicklable(hkl.load(os.path.join(
                spath, '%05d.hkl' % i)))
            assert fp_i.ip_scores.size == 500
            confmat[i, i] = 0
            for j in range(i + 1, n):
                fp_j = system.forwardPassFromHicklable(hkl.load(os.path.join(
                    spath, '%05d.hkl' % j)))
                pair = wrapper.makePair([i, j])
                if pair.imname(0) not in stereo_cache:
                    stereo_images = [pair.im[0], pair.rim_0]
                    stereo_cache[pair.imname(0)] = p3p.pointsFromStereo(
                        stereo_images, fp_i.ips_rc, pair.K, pair.baseline)
                least_num = evaluate.leastNumForKInliers(
                    pair, [fp_i, fp_j], 20, stereo_cache=stereo_cache)
                confmat[i, j] = least_num
                confmat[j, i] = least_num
                if least_num == 500:
                    break
                least_nums.append(least_num)
                print('\t%05d: %d' % (j, least_num))
        hkl.dump([confmat, least_nums], open(
            hyperparams.resultPath() + '_confmat_%s.hkl' % seq.name(), 'w'))


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    pr.dump_stats('%s.profile' % __file__)
