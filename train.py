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
import cv2
import numpy as np
import tensorflow as tf
import time

from rpg_common_py.trainer import Trainer

import sips2.augment as augment
import sips2.evaluate as evaluate
import sips2.flags as flags
import sips2.graph as graph
import sips2.hyperparams as hyperparams
import sips2.multiscale as multiscale
import sips2.plot_utils as plot_utils
import sips2.system as system

FLAGS = flags.FLAGS


def makeBatchAndInlierLabels(image, forward_pass, inlier_indices):
    batch = forward_pass.getPatchesSortedByScore(image, FLAGS.d)

    if FLAGS.debug_plot:
        score_patches = forward_pass.getPatchesSortedByScore(None, FLAGS.d)
        assert score_patches.shape[0] == 500
        im = plot_utils.tile(score_patches, 25, 20)
        cv2.imwrite('debug/patches_%d.ppm' % time.time(), im)

    inliers = np.zeros(batch.shape[0], dtype=bool)
    inliers[inlier_indices] = True
    
    return batch, inliers
    

def makeBatchAndInlierLabelsDouble(dp, fps, matched_indices, inliers):
    batch0, inl0 = makeBatchAndInlierLabels(
            dp.im[0], fps[0], matched_indices[0][inliers])
    batch1, inl1 = makeBatchAndInlierLabels(
            dp.im[1], fps[1], matched_indices[1][inliers])
    
    return np.concatenate((batch0, batch1)), np.concatenate((inl0, inl1))
    

def run():
    hyperparams.announceTraining()

    g = graph.Graph()
    train_pairs = hyperparams.getTrainDataGen()
    if FLAGS.tds == 'hp':
        tval_pairs = train_pairs.subSampled(5)
    eval_pairs = hyperparams.getEvalDataGen()
    
    tf_sess = tf.Session()

    forward_passer = hyperparams.getForwardPasser(g, tf_sess)
    
    def step(sess):
        dp = train_pairs.getRandomDataPoint()
        if FLAGS.augment and FLAGS.tds == 'hp':
            dp = augment.augmentHpatchPair(dp)
        # Feedforward
        fps = [forward_passer(im) for im in dp.im]
        if type(fps[0]) == multiscale.ForwardPass:
            flat = [i.toFlatForwardPass() for i in fps]
        else:
            flat = fps
        matched_indices = system.match(flat)
        inliers = system.getInliers(dp, flat, matched_indices)
        print('%d inliers' % np.count_nonzero(inliers))
        batch, inlier_labels = makeBatchAndInlierLabelsDouble(
                dp, fps, matched_indices, inliers)
        sess.run(g.train_step,
                 feed_dict={g.train_input: batch, g.inlier_mask: inlier_labels})
    
    def validate(_):
        if FLAGS.tds == 'hp':
            return [evaluate.succinctness(eval_pairs, forward_passer, 10),
                    evaluate.succinctness(tval_pairs, forward_passer, 10)]
        else:
            return [evaluate.succinctness(eval_pairs, forward_passer, 10)]
    
    trainer = Trainer(step, validate, hyperparams.checkpointRoot(),
                      hyperparams.shortString(),
                      check_every=FLAGS.val_every, best_index=0,
                      best_is_max=True)
    trainer.trainUpToNumIts(tf_sess, FLAGS.its)


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    
    run()
    
    pr.disable()
    pr.dump_stats('train.profile')
