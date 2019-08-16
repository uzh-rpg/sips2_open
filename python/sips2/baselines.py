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
import numpy as np
import os

import rpg_datasets_py.utils.symlink
import rpg_datasets_py.hpatches
import rpg_datasets_py.euroc

import hyperparams
import sequences
import system

FLAGS = absl.flags.FLAGS


class OpenCVForwardPasser(object):
    def __init__(self, type_string):
        if type_string == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create()
        elif type_string == 'sift':
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif type_string == 'orb':
            self.detector = cv2.ORB_create()
        else:
            assert False

    def __call__(self, image):
        ips_descs = self.detector.detectAndCompute(image, None)
        # TODO subpixel support (problem with stereo)
        ips_rc = np.array([[i.pt[1], i.pt[0]] for i in ips_descs[0]]).astype(
            'int')
        scores = np.array([i.response for i in ips_descs[0]])
        descrs = ips_descs[1]

        resort = np.argsort(-scores)
        ips_rc = ips_rc[resort, :]
        scores = scores[resort]
        descrs = descrs[resort, :]

        return system.ForwardPass(ips_rc.T, scores, descrs)


def subfolderNames(data_gen):
    if type(data_gen) == rpg_datasets_py.hpatches.HPatches:
        return data_gen.folder_names
    else:
        assert type(data_gen) == sequences.FixPairs
        if FLAGS.ds == 'kt':
            suffix = '/image_0'
        elif FLAGS.ds in ['eu', 'eumh']:
            suffix = '/rect_0'
        result = set()
        for pair in data_gen:
            result.add(pair.seqname + suffix)
        return list(result)


def inNameForDataset():
    if FLAGS.ds == 'hp':
        return 'min_hpatches'
    elif FLAGS.ds == 'kt':
        return 'kitti'
    elif FLAGS.ds in ['eu', 'eumh']:
        return 'euroc'
    else:
        assert False


def outNameForDataset():
    if FLAGS.ds == 'hp':
        return 'min_hpatches_out'
    elif FLAGS.ds == 'kt':
        return 'kitti_out'
    elif FLAGS.ds in ['eu', 'eumh']:
        return 'euroc_out'
    else:
        assert False


def printLFNetCommands():
    data_gen = hyperparams.getEvalDataGen()
    for k in [30, 50, 100, 150]:
        for folder in subfolderNames(data_gen):
            print('python run_lfnet.py --top_k=%d --in_dir=%s/'
                  '%s --out_dir=%s/%d/%s' %
                  (k, inNameForDataset(), folder,
                   outNameForDataset(), k, folder))


def printSuperPointCommands():
    data_gen = hyperparams.getEvalDataGen()
    if FLAGS.ds == 'hp':
        glob = '*.pgm'
    elif FLAGS.ds in ['eu', 'eumh']:
        glob = '*.png'
    else:
        assert FLAGS.ds == 'kt'
        glob = '*.png'
    for folder in subfolderNames(data_gen):
        print('python superpoint_frontend_parse.py %s/%s --cuda'
              ' --write_dir=%s/%s --img_glob=\'%s\'' %
              (inNameForDataset(), folder, outNameForDataset(), folder, glob))


def lfNetOutDir(n_extracted_pts):
    return os.path.join(
        os.path.dirname(__file__), 'baseline_links', 'lf-net-release',
        outNameForDataset(), '%d' % n_extracted_pts)


def superOutDir():
    return os.path.join(
        os.path.dirname(__file__), 'baseline_links',
        'SuperPointPretrainedNetwork', outNameForDataset())


class ScorelessForwardPass(object):
    def __init__(self, ips_rc, descriptors, scales):
        assert ips_rc.shape[0] == 2
        assert ips_rc.shape[1] == descriptors.shape[0]
        assert ips_rc.shape[1] == scales.shape[0]
        #TODO support float?
        self.ips_rc = ips_rc.astype(int)
        self.descriptors = descriptors
        self.scales = scales

    def __getitem__(self, slc):
        # Only applies for subselection; matched forward passes are ok to have
        # non-ordered score.
        if type(self.scales) == np.ndarray:
            scales = self.scales[slc]
        else:
            scales = self.scales
        return ScorelessForwardPass(
            self.ips_rc[:, slc], self.descriptors[slc, :], scales)


def parseLFNetOuts(data_gen, n_extracted_pts):
    forward_passes = dict()
    if FLAGS.ds == 'eu':
        eurocs = rpg_datasets_py.euroc.EurocSeq('V1_01_easy')
    elif FLAGS.ds == 'eumh':
        eurocs = rpg_datasets_py.euroc.EurocSeq('MH_01_easy')
    # TODO consolidate with SuperPoint!
    if type(data_gen) == rpg_datasets_py.hpatches.HPatches:
        for folder_name in data_gen.folder_names:
            path = os.path.join(lfNetOutDir(n_extracted_pts), folder_name)
            assert os.path.exists(path)
            for i in range(1, 7):
                npz_path = os.path.join(path, '%d.pgm.npz' % i)
                im_path = os.path.join(rpg_datasets_py.utils.symlink.symlink(
                    'min_hpatches'), folder_name, '%d.pgm' % i)
                im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                if im.shape[0] > im.shape[1] and im.shape[0] > 640:
                    kp_factor = float(im.shape[0]) / 640.
                elif im.shape[1] > 640:
                    kp_factor = float(im.shape[1]) / 640.
                else:
                    kp_factor = 1
                lf_out = np.load(npz_path)
                forward_pass = ScorelessForwardPass(
                    np.fliplr(lf_out['kpts']).T * kp_factor, lf_out['descs'],
                    lf_out['scales'])
                forward_passes['%s%d' % (folder_name, i)] = forward_pass
    else:
        assert type(data_gen) == sequences.FixPairs
        for pair in data_gen:
            assert pair.im[0].shape[1] > pair.im[0].shape[0]
            assert pair.im[0].shape[1] > 640
            assert pair.im[0].shape[1] == pair.im[1].shape[1]
            kp_factor = float(pair.im[0].shape[1]) / 640.
            if FLAGS.ds == 'kt':
                suffix = 'image_0'
            elif FLAGS.ds in ['eu', 'eumh']:
                suffix = 'rect_0'
            path = os.path.join(lfNetOutDir(n_extracted_pts),
                                pair.seqname, suffix)
            if not os.path.exists(path):
                print(FLAGS.ds)
                raise Exception('Missing LFNet runs at %s' % path)
            for i in [0, 1]:
                index = pair.indices[i]
                if FLAGS.ds not in ['eu', 'eumh']:
                    npz_path = os.path.join(path, '%06d.png.npz' % index)
                else:
                    imname = os.path.basename(eurocs.images[index])
                    npz_path = os.path.join(path, '%s.npz' % imname)

                lf_out = np.load(npz_path)
                forward_pass = ScorelessForwardPass(
                    np.fliplr(lf_out['kpts']).T * kp_factor, lf_out['descs'],
                    lf_out['scales'])
                forward_passes['%s%d' % (pair.seqname, index)] = forward_pass
    return forward_passes


def forwardPassFromSuperHkl(hkl_path):
    super_out = hkl.load(open(hkl_path, 'r'))
    return system.ForwardPass(
        np.fliplr(super_out[1][:, :2]).T.astype(int), super_out[1][:, 2],
        super_out[1][:, 3:].astype(np.float32), 1., super_out[0])


def parseSuperPointOuts(data_gen):
    if FLAGS.ds == 'eu':
        eurocs = rpg_datasets_py.euroc.EurocSeq('V1_01_easy')
    elif FLAGS.ds == 'eumh':
        eurocs = rpg_datasets_py.euroc.EurocSeq('MH_01_easy')
    forward_passes = dict()
    if type(data_gen) == rpg_datasets_py.hpatches.HPatches:
        for folder_name in data_gen.folder_names:
            path = os.path.join(superOutDir(), folder_name)
            assert os.path.exists(path)
            for i in range(1, 7):
                hkl_path = os.path.join(path, '%d.pgm.hkl' % i)
                forward_passes['%s%d' % (folder_name, i)] = \
                    forwardPassFromSuperHkl(hkl_path)
    else:
        assert type(data_gen) == sequences.FixPairs
        for pair in data_gen:
            if FLAGS.ds == 'kt':
                suffix = 'image_0'
            elif FLAGS.ds in ['eu', 'eumh']:
                suffix = 'rect_0'
            path = os.path.join(superOutDir(), pair.seqname, suffix)
            if not os.path.exists(path):
                raise Exception('Missing SuperPoint runs at %s' % path)
            for i in [0, 1]:
                index = pair.indices[i]

                if FLAGS.ds not in ['eu', 'eumh']:
                    hkl_path = os.path.join(path, '%06d.png.hkl' % index)
                else:
                    imname = os.path.basename(eurocs.images[index])
                    hkl_path = os.path.join(path, '%s.hkl' % imname)

                forward_passes['%s%d' % (pair.seqname, index)] = \
                    forwardPassFromSuperHkl(hkl_path)

    return forward_passes
