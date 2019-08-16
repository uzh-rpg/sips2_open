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
import cv2
import hickle as hkl
import os

import sips2.flags
import sips2.hyperparams as hyperparams

FLAGS = absl.flags.FLAGS


def process(spath, irange, seq, fpasser):
    fps = fpasser.parallelForward(
        [cv2.imread(seq.images[i], cv2.IMREAD_GRAYSCALE) for i in irange])
    for i in irange:
        print('%d/%d' % (i, len(seq.images)))
        im = seq.images[i]
        path = os.path.join(spath, '%05d' % i)
        fp = fps[i - irange[0]]
        rendering = fp.render()
        if FLAGS.debug_plot:
            cv2.imshow('render', rendering)
            cv2.waitKey(1)
        cv2.imwrite(path + '.jpg', rendering)
        hkl.dump(fp.hicklable(), open(path + '.hkl', 'w'))


if __name__ == '__main__':
    seqs = hyperparams.getEvalSequences()
    g, sess = hyperparams.modelFromCheckpoint()
    fpasser = hyperparams.getForwardPasser(g, sess)
    for seq in seqs:
        print(seq.name())
        spath = os.path.join(hyperparams.seqFpsPath(), seq.name())
        if not os.path.exists(spath):
            os.makedirs(spath)
        n = len(seq.images)
        for i in range(0, n, FLAGS.fpbs):
            process(spath, range(i, i + FLAGS.fpbs), seq, fpasser)
        process(spath, range(n - (n % FLAGS.fpbs), n), seq, fpasser)
