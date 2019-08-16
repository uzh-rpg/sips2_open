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

from absl import flags
import os
import tensorflow as tf

from rpg_datasets_py.hpatches import HPatches
import rpg_datasets_py.euroc
import rpg_datasets_py.kitti
import rpg_datasets_py.robotcar
import rpg_datasets_py.tum_mono

import baselines
import graph
import multiscale
import sequences
import system

FLAGS = flags.FLAGS


def shortString():
    ret = 'd=%d_tds=%s_nms=%d' % (FLAGS.d, FLAGS.tds, FLAGS.nms)
    if FLAGS.num_scales > 1:
        ret = ret + '_ms%d_%.02f' % (FLAGS.num_scales, FLAGS.scale_factor)
    if FLAGS.pbs:
        ret = ret + '_pbs'
    if FLAGS.augment:
        ret = ret + '_aug'
    if FLAGS.w != 128:
        ret = ret + '_w=%d' % FLAGS.w
    if FLAGS.scale_aug_range != 1.:
        ret = ret + '_sar=%.01f' % FLAGS.scale_aug_range
    if FLAGS.klti:
        ret = ret + '_klti'
    if FLAGS.lk:
        ret = ret + '_lk'
    if FLAGS.ol != 0.5:
        ret = ret + '_ol=%.2f' % FLAGS.ol
    return ret


def announce(what):
    print(os.linesep + os.linesep + what + os.linesep + os.linesep)


def announceTraining():
    announce('Training %s...' % shortString())


def getTrainDataGen():
    if FLAGS.klti:
        pair_class = sequences.PairWithIntermediates
    else:
        pair_class = sequences.PairWithStereo
    assert FLAGS.pck == 'tr'
    if FLAGS.tds == 'kt':
        ks = [rpg_datasets_py.kitti.KittiSeq(i)
              for i in rpg_datasets_py.kitti.split('training')]
        return sequences.MultiSeqTrackingPairPicker(ks, FLAGS.ol, pair_class)
    elif FLAGS.tds == 'rc':
        rs = rpg_datasets_py.robotcar.getSplitSequences('training')
        return sequences.MultiSeqTrackingPairPicker(rs, FLAGS.ol, pair_class)
    elif FLAGS.tds == 'tm':
        tms = [rpg_datasets_py.tum_mono.Sequence(i)
               for i in ['01', '02', '03', '48', '49', '50']]
        return sequences.MultiSeqTrackingPairPicker(
            tms, FLAGS.ol, sequences.PairWithIntermediates)
    elif FLAGS.tds == 'tmb':
        tms = [rpg_datasets_py.tum_mono.Sequence('%02d' % i)
               for i in range(1, 51)]
        return sequences.MultiSeqTrackingPairPicker(
            tms, FLAGS.ol, sequences.PairWithIntermediates)
    elif FLAGS.tds == 'tmbrc':
        tms = [rpg_datasets_py.tum_mono.Sequence('%02d' % i)
               for i in range(1, 51)] + \
              [rpg_datasets_py.robotcar.CroppedGraySequence(
                  '2014-07-14-14-49-50')]
        return sequences.MultiSeqTrackingPairPicker(
            tms, FLAGS.ol, sequences.PairWithIntermediates)
    elif FLAGS.tds == 'en':
        ks = [rpg_datasets_py.kitti.KittiSeq(i)
              for i in rpg_datasets_py.kitti.split('training')]
        rs = rpg_datasets_py.robotcar.getSplitSequences('training')
        tms = [rpg_datasets_py.tum_mono.Sequence(i)
               for i in ['01', '02', '03', '48', '49', '50']]
        pair_classes = [pair_class] * len(ks) + \
                       [sequences.PairWithIntermediates] * (len(rs) + len(tms))
        return sequences.MultiSeqTrackingPairPicker(
            ks + rs + tms, FLAGS.ol, pair_classes)
    elif FLAGS.tds == 'hp':
        data_gen = HPatches('training', use_min=True)
    else:
        assert False
    
    return data_gen


def getEvalDataGen():
    if not FLAGS.testing:
        if FLAGS.ds == 'kt':
            val_seqs = rpg_datasets_py.kitti.split('validation')
            assert len(val_seqs) == 1
            k = rpg_datasets_py.kitti.KittiSeq(val_seqs[0])
            rpick = sequences.TrackingPairPicker(
                k, 0.5, pair_class=sequences.PairWithStereo)
            return sequences.FixPairs(rpick, 100)
        else:
            assert FLAGS.ds == 'hp'
            ret = HPatches('validation', use_min=True)
            print(ret.folder_names)
            return ret
    else:
        if FLAGS.ds == 'hp':
            ret = HPatches('testing', use_min=True)
            print(ret.folder_names)
            return ret
        elif FLAGS.ds == 'kt':
            val_seqs = rpg_datasets_py.kitti.split('testing')
            assert len(val_seqs) == 1
            k = rpg_datasets_py.kitti.KittiSeq(val_seqs[0])
            rpick = sequences.TrackingPairPicker(
                k, 0.5, pair_class=sequences.PairWithStereo)
            return sequences.FixPairs(rpick, 100)
        elif FLAGS.ds == 'eu':
            seq = rpg_datasets_py.euroc.EurocSeq('V1_01_easy')
            rpick = sequences.TrackingPairPicker(
                seq, 0.5, pair_class=sequences.PairWithStereo)
            return sequences.FixPairs(rpick, 100)
        elif FLAGS.ds == 'eumh':
            seq = rpg_datasets_py.euroc.EurocSeq('MH_01_easy')
            rpick = sequences.TrackingPairPicker(
                seq, 0.5, pair_class=sequences.PairWithStereo)
            return sequences.FixPairs(rpick, 100)
        else:
            raise NotImplementedError


def getEvalSequences():
    if FLAGS.ds == 'kt':
        if FLAGS.testing:
            seq_names = rpg_datasets_py.kitti.split('testing')
        else:
            seq_names = rpg_datasets_py.kitti.split('validation')
        return [rpg_datasets_py.kitti.KittiSeq(i) for i in seq_names]
    elif FLAGS.ds == 'eumh':
        assert FLAGS.testing
        seq_names = ['MH_01_easy']
        return [rpg_datasets_py.euroc.EurocSeq(i) for i in seq_names]
    else:
        assert FLAGS.ds == 'eu'
        assert FLAGS.testing
        seq_names = ['V1_01_easy']
        return [rpg_datasets_py.euroc.EurocSeq(i) for i in seq_names]


def getForwardPasser(graph=None, sess=None):
    if FLAGS.baseline == '':
        assert graph is not None
        assert sess is not None
        fp = system.ForwardPasser(
                graph, sess, FLAGS.num_test_pts, FLAGS.nms)
        if FLAGS.num_scales > 1:
            return multiscale.ForwardPasser(
                fp, FLAGS.scale_factor, FLAGS.num_scales)
        else:
            return fp
    elif FLAGS.baseline in ['surf', 'sift']:
        return baselines.OpenCVForwardPasser(FLAGS.baseline)
    else:
        raise Exception('Baseline %s unknown' % FLAGS.baseline)


def modelFromCheckpoint():
    if FLAGS.baseline is not '':
        return None, None
    tf.reset_default_graph()
    g = graph.Graph()
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, checkpointPath())
    return g, sess


def checkpointRoot():
    return os.path.join(os.path.dirname(__file__), 'checkpoints')


def checkpointPath():
    if FLAGS.val_best:
        return os.path.join(checkpointRoot(), shortString() + '_best')
    else:
        return os.path.join(checkpointRoot(), shortString())


def trainStatsPath():
    return '%s_stats' % checkpointPath()


def methodString():
    if FLAGS.baseline == '':
        return shortString()
    else:
        return FLAGS.baseline


def evalString(k=True):
    result = '%s_%d' % (FLAGS.ds, FLAGS.num_test_pts)
    if k:
        result += '_%d' % FLAGS.k
    if FLAGS.testing:
        result += '_TESTING'
    return result


def vt():
    if FLAGS.testing:
        return 'testing'
    else:
        return 'validation'


def methodEvalString(k=True):
    return '%s_%s' % (methodString(), evalString(k=k))


def announceEval():
    announce('Evaluating %s...' % methodEvalString())
    if FLAGS.testing:
        announce('THIS IS A TESTING RUN!!!')


def resultPath(k=True):
    return os.path.join('results', methodEvalString(k=k))


def wasInlierFilePath():
    return '%s_wasinl' % resultPath()


def cachedForwardPath():
    return '%s_cached_fp.hkl' % resultPath(k=False)


def evalPath():
    return '%s_n_r_t.hkl' % resultPath()


def choutRootPath():
    return '%s_chouts' % resultPath()


def seqFpsPath():
    here = os.path.dirname(__file__)
    return os.path.join(here, 'sequence_fps', methodEvalString(k=False))


def label():
    if FLAGS.baseline != '':
        return FLAGS.baseline
    else:
        if FLAGS.num_scales > 1:
            return 'ours, multi-scale'
        else:
            return 'ours, single scale'
