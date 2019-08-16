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
import cachetools
import cv2
import numpy as np
import skimage.measure

import rpg_datasets_py.hpatches

import plot_utils
import p3p
import sequences

FLAGS = absl.flags.FLAGS


def sortedInterestPointsRowCol(image_scores, num, nonmax_sup_size,
                               border_margin):
    if nonmax_sup_size > 0:
        dilation = cv2.dilate(
            image_scores, np.ones((nonmax_sup_size, nonmax_sup_size)))
    else:
        dilation = image_scores
    nonmaxes = image_scores == dilation
    indices = np.nonzero(nonmaxes)

    # Respect border margin
    bad_row = np.logical_or(
        indices[0] < border_margin,
        indices[0] >= image_scores.shape[0] - border_margin)
    bad_col = np.logical_or(
        indices[1] < border_margin,
        indices[1] >= image_scores.shape[1] - border_margin)
    good = np.logical_not(np.logical_or(bad_row, bad_col))
    indices = tuple([indices[0][good], indices[1][good]])

    nonmax_scores = image_scores[indices]
    if nonmax_scores.size < num:
        sup_diam = 2 * nonmax_sup_size + 1
        unsurprising_less = num * sup_diam * sup_diam
        if image_scores.shape[0] * image_scores.shape[1] > unsurprising_less:
            print('Warning:')
            print('Nonmax score size %d is less than desired count %d' %
                  (nonmax_scores.size, num))
            print('Image resolution has been %d %d' %
                  (image_scores.shape[0], image_scores.shape[1]))
            print('Reducing amount of returned points!')
        num = nonmax_scores.size
    sort_indices = np.argsort(-nonmax_scores)

    return np.array([indices[0][sort_indices[:num]], indices[1][
        sort_indices[:num]]]), nonmax_scores[sort_indices[:num]]


def withinRadiusOfBorder(point_rc, image_shape, radius):
    below = np.any(point_rc < radius)
    above = np.any(point_rc >= np.array(image_shape) - radius)
    return below or above


class ForwardPass(object):
    def __init__(self, ips_rc, ip_scores, descriptors, scales=1.,
                 image_scores=None):
        assert descriptors.shape[0] == ips_rc.shape[1]
        self.ips_rc = ips_rc
        assert ips_rc.dtype == int
        self.ip_scores = ip_scores
        self.descriptors = descriptors
        self.scales = scales
        if type(self.scales) == np.ndarray:
            assert len(scales) == ips_rc.shape[1]
        self.image_scores = image_scores

    def __getitem__(self, slc):
        # Only applies for subselection; matched forward passes are ok to have
        # non-ordered score.
        assert np.all(self.ip_scores[:-1] >= self.ip_scores[1:])
        if type(self.scales) == np.ndarray:
            scales = self.scales[slc]
        else:
            scales = self.scales
        return ForwardPass(self.ips_rc[:, slc], self.ip_scores[slc],
                           self.descriptors[slc, :], scales, self.image_scores)

    def __len__(self):
        return len(self.ip_scores)

    def reducedTo(self, num_ips):
        return self[:num_ips]

    def hicklable(self):
        return [self.ips_rc, self.ip_scores, self.descriptors, self.scales]

    def getPatchesSortedByScore(self, image, radius):
        assert np.all(self.ip_scores[:-1] >= self.ip_scores[1:])
        assert self.scales == 1.
        n = self.ips_rc.shape[1]
        diam = radius * 2 + 1
        batch = np.zeros([n, diam, diam])

        for i in range(n):
            if withinRadiusOfBorder(self.ips_rc[:, i], image.shape, radius):
                print('Invalid interest point at:')
                print(self.ips_rc[:, i])
                print('All IPs are:')
                print(self.ips_rc)
                cv2.imwrite('debug/bad_ip.ppm',
                            plot_utils.colored(self.image_scores))
                assert False
            batch[i, :, :] = \
                image[self.ips_rc[0, i] - radius:self.ips_rc[0, i] + radius + 1,
                      self.ips_rc[1, i] - radius:self.ips_rc[1, i] + radius + 1]

        return batch

    def render(self, n_max=0, fallback_im=None):
        if self.image_scores is not None:
            im = cv2.applyColorMap((self.image_scores * 255).astype(np.uint8),
                                   cv2.COLORMAP_JET)
        else:
            assert fallback_im is not None
            im = cv2.cvtColor(fallback_im, cv2.COLOR_GRAY2BGR)

        if n_max == 0:
            n_max = self.ips_rc.shape[1]
        for i in range(n_max):
            thickness_relevant_score = \
                np.clip(self.ip_scores[i], 0.2, 0.6) - 0.2
            thickness = int(thickness_relevant_score * 20)
            if type(self.scales) == np.ndarray:
                radius = int(self.scales[i] * 10)
            else:
                radius = 10
            cv2.circle(im, tuple(self.ips_rc[[1, 0], i]),
                       radius, (0, 255, 0), thickness, cv2.LINE_AA)
        return im


def forwardPassFromHicklable(hicklable):
    return ForwardPass(hicklable[0], hicklable[1], hicklable[2],
                       scales=hicklable[3])


class ForwardPasser(object):
    def __init__(self, graph, sess, num_ips, nms_size):
        self._graph = graph
        self._sess = sess
        self.num_ips = num_ips
        self.nms_size = nms_size
        self._descriptor_extractor = cv2.xfeatures2d.SURF_create()

    def forwardPassFromScores(self, image, image_scores):
        ips_rc, ip_scores = sortedInterestPointsRowCol(
            image_scores, self.num_ips, self.nms_size, FLAGS.d)
        cv_ips = [cv2.KeyPoint(
            x=ips_rc[1, i], y=ips_rc[0, i], _size=31, _octave=0,
            _angle=0., _response=0.) for i in range(ips_rc.shape[1])]
        _, descriptors = self._descriptor_extractor.compute(image, cv_ips)
        return ForwardPass(ips_rc, ip_scores, descriptors, 1., image_scores)

    def __call__(self, image):
        batch = np.expand_dims(image, 0)
        image_scores = self._sess.run(
            self._graph.ff_output, feed_dict={self._graph.ff_input: batch})
        image_scores = np.squeeze(image_scores)
        assert np.all(image.shape == image_scores.shape)
        return self.forwardPassFromScores(image, image_scores)

    def parallelForward(self, images):
        batch = np.array(images)
        images_scores = self._sess.run(
            self._graph.ff_output, feed_dict={self._graph.ff_input: batch})
        return [self.forwardPassFromScores(images[i], images_scores[i])
                for i in range(len(images))]


class ForwardPassCache(cachetools.Cache):
    def __init__(self, forward_passer):
        self._forward_passer = forward_passer
        cachetools.Cache.__init__(self, 1000000)

    def __getitem__(self, image):
        coded = tuple([image.shape[1]] + image.ravel().tolist())
        return cachetools.Cache.__getitem__(self, coded)

    def __missing__(self, key):
        width = key[0]
        image = np.reshape(np.array(key[1:], dtype=np.uint8), [-1, width])
        result = self._forward_passer(image)
        self[key] = result
        return result


# Keeping things simple for now: No Gaussians, naive downscaling with
# well-defined rules.
def downScaled(image):
    divisible = image[:(2 * (image.shape[0] / 2)), :(2 * (image.shape[1] / 2))]
    assert divisible.shape[0] % 2 == 0
    assert divisible.shape[1] % 2 == 0
    return skimage.measure.block_reduce(divisible, (2, 2), np.mean)


def match(forward_passes):
    if FLAGS.baseline == 'orb':
        print('Using Hamming matching for ORB...')
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(forward_passes[0].descriptors,
                            forward_passes[1].descriptors)
    return [np.array([x.queryIdx for x in matches]),
            np.array([x.trainIdx for x in matches])]


def getHpatchInliers(pair, matched_fps):
    assert type(pair) == rpg_datasets_py.hpatches.HPatchPair
    corresps1 = np.flipud(pair.correspondences(
        np.flipud(matched_fps[0].ips_rc)))
    dists = np.linalg.norm(matched_fps[1].ips_rc - corresps1, axis=0)
    mult = np.maximum(matched_fps[0].scales, matched_fps[1].scales)
    inlier_mask = dists < (3 * mult)
    if FLAGS.debug_plot:
        print('%d inliers' % np.count_nonzero(inlier_mask))
    return inlier_mask


def getInliers(pair, forward_passes, matched_indices, stereo_cache=None):
    matched_fps = [forward_passes[i][matched_indices[i]] for i in [0, 1]]
    if type(pair) == rpg_datasets_py.hpatches.HPatchPair:
        matched_mask = getHpatchInliers(pair, matched_fps)
    elif type(pair) == sequences.PairWithStereo:
        matched_mask = p3p.ransac(
            pair, matched_fps, stereo_cache=stereo_cache)
    else:
        assert type(pair) == sequences.PairWithIntermediates
        matched_mask = pair.getInliers([fp.ips_rc for fp in matched_fps])
    return matched_mask

