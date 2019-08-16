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
import copy
import cv2
import numpy as np
import os

import hyperparams
import system

FLAGS = absl.flags.FLAGS


class Tiling(object):
    def __init__(self, images):
        self.x_offsets = [0]
        self.y_shapes = []
        for image in images:
            self.x_offsets.append(self.x_offsets[-1] + image.shape[1])
            self.y_shapes.append(image.shape[0])
        if not np.all(np.array(self.y_shapes[:-1]) >
                      np.array(self.y_shapes[1:])):
            print(self.y_shapes)
            assert False

    def tile(self, images):
        im = np.zeros([self.y_shapes[0], self.x_offsets[-1]])
        n = len(images)
        assert n == len(self.y_shapes)
        for i in range(n):
            im[:self.y_shapes[i], self.x_offsets[i]:self.x_offsets[i + 1]] = \
                images[i]
        return im

    def untile(self, tiled_image):
        n = len(self.y_shapes)
        images = []
        for i in range(n):
            images.append(tiled_image[:self.y_shapes[i],
                          self.x_offsets[i]:self.x_offsets[i+1]])
        return images

    def hicklable(self):
        return [self.x_offsets, self.y_shapes]


def tilingFromHicklable(hicklable):
    t = Tiling([])
    t.x_offsets = hicklable[0]
    t.y_shapes = hicklable[1]
    return t


class Pyramid(object):
    def __init__(self, image, factor, size):
        self.images = [image]
        for _ in range(1, size):
            self.images.append(cv2.resize(
                self.images[-1], (0,0), fx=factor, fy=factor))
        self.tiling = Tiling(self.images)


class ForwardPass(object):
    def __init__(self, factor, forward_passes, tiling):
        self.forward_passes = forward_passes
        self.factor = factor
        self.tiling = tiling

    def locationInAllLayers(self, level_index, point_index):
        if level_index >= len(self.forward_passes):
            raise Exception('Level index %d exceeds level count %d' %
                            (level_index, len(self.forward_passes)))
        result = np.zeros([2, len(self.forward_passes)])
        self_location = self.forward_passes[level_index].ips_rc[:, point_index]
        for i in range(len(self.forward_passes)):
            result[:, i] = self_location * (self.factor ** (i - level_index))

        return result

    def overlaps(self, ids, level_index, point_index, nms):
        if level_index >= len(self.forward_passes):
            raise Exception('Level index %d exceeds level count %d' %
                            (level_index, len(self.forward_passes)))
        suppressor_locations = self.locationInAllLayers(
            level_index, point_index)
        for level_i in range(len(self.forward_passes)):
            i_ids = ids[0, ids[1, :] == level_i]
            suppressee_locations = self.forward_passes[level_i].ips_rc[:, i_ids]
            diff = suppressee_locations.T - suppressor_locations[:, level_i]
            sup_row = np.abs(diff[:, 0]) < nms
            sup_col = np.abs(diff[:, 1]) < nms
            overlap = np.logical_and(sup_row, sup_col)
            if np.any(overlap):
                return True
        return False

    def crossScaleNms(self, ids, num_ips, nms):
        """ ids: 2xn, first row: id in level, second row: id of level """
        result = - np.ones([2, num_ips], dtype=int)
        num_valid = 0
        n = ids.shape[1]
        for i in range(n):
            if not self.overlaps(result, ids[1, i], ids[0, i], nms):
                result[:, num_valid] = ids[:, i]
                num_valid = num_valid + 1
                if num_valid == num_ips:
                    break
        if num_valid < num_ips:
            print('Warning: Not able to extract %d points, got only %d!' %
                  (num_ips, num_valid))
            result = result[:, :num_valid]
        return result

    def selectTop(self, num_ips, nms_size):
        score_aggregate = []
        id_aggregate = []
        for i in range(len(self.forward_passes)):
            fpass = self.forward_passes[i]
            scores = fpass.ip_scores
            score_aggregate.append(scores)
            id_aggregate.append(np.vstack((np.arange(scores.size),
                                           i * np.ones(scores.size))))
        scores = np.hstack(score_aggregate)
        ids = np.hstack(id_aggregate).astype(int)
        sorting = np.argsort(-scores)
        ids = ids[:, sorting]
        if nms_size is not None:
            ids = self.crossScaleNms(ids, num_ips, nms_size)
        else:
            ids = ids[:, :num_ips]
        for i in range(len(self.forward_passes)):
            i_ids = ids[0, ids[1, :] == i]
            self.forward_passes[i] = self.forward_passes[i][i_ids]

    def reducedTo(self, num_ips):
        ret = copy.deepcopy(self)
        ret.selectTop(num_ips, None)
        return ret

    def ipsInTiledImage(self):
        ips_in_tiled = []
        for fp_i in range(0, len(self.forward_passes)):
            fp = self.forward_passes[fp_i]
            for ip_i in range(fp.ips_rc.shape[1]):
                ips_in_tiled.append(
                    fp.ips_rc[:, ip_i] +
                    np.array([0, self.tiling.x_offsets[fp_i]]))
        return np.array(ips_in_tiled).T

    def flatSorting(self):
        return np.argsort(-np.hstack(
            [fp.ip_scores for fp in self.forward_passes]))

    def ipsInTiledImageWithFlatIndices(self, indices):
        return self.ipsInTiledImage()[:, self.flatSorting()[indices]]

    def toFlatForwardPass(self, get_tile_ips=False):
        # Collect interest points.
        flat_ips_rc = [self.forward_passes[i].ips_rc.astype(float) /
                       (self.factor ** i)
                       for i in range(len(self.forward_passes))]
        # Collect scaling factors.
        scales = [
            np.ones_like(self.forward_passes[i].ip_scores) / (self.factor ** i)
            for i in range(len(self.forward_passes))]
        # Generate tiled score image.
        if self.forward_passes[0].image_scores is not None:
            tiled = self.tiling.tile(
                [fp.image_scores for fp in self.forward_passes])
        else:
            tiled = None
        unsorted_ips = np.hstack(flat_ips_rc).astype(int)
        unsorted_scores = np.hstack(
            [fp.ip_scores for fp in self.forward_passes])
        unsorted_descs = np.vstack(
            [fp.descriptors for fp in self.forward_passes])
        unsorted_scales = np.hstack(scales)
        unsorted_tile_ips = self.ipsInTiledImage()

        assert unsorted_ips.shape[1] == unsorted_scores.size
        assert unsorted_scores.size == unsorted_descs.shape[0]
        assert unsorted_descs.shape[0] == unsorted_scales.size
        assert unsorted_scales.size == unsorted_tile_ips.shape[1]

        sorting = np.argsort(-unsorted_scores)
        if get_tile_ips:
            return system.ForwardPass(
                unsorted_ips[:, sorting], unsorted_scores[sorting],
                unsorted_descs[sorting, :], unsorted_scales[sorting], tiled), \
                unsorted_tile_ips[:, sorting]
        else:
            return system.ForwardPass(
                unsorted_ips[:, sorting], unsorted_scores[sorting],
                unsorted_descs[sorting, :], unsorted_scales[sorting], tiled)

    def debugImage(self):
        ffp, _ = self.toFlatForwardPass()
        im = cv2.applyColorMap((ffp.image_scores * 255).astype(np.uint8),
                               cv2.COLORMAP_JET)
        for i in range(ffp.ips_rc.shape[1]):
            cv2.circle(im, tuple(ffp.ips_rc[[1, 0], i]),
                       int(ffp.scales[i] * 5), (0, 255, 0), 1, cv2.LINE_AA)
        for fp_i in range(1, len(self.forward_passes)):
            fp = self.forward_passes[fp_i]
            for ip_i in range(fp.ips_rc.shape[1]):
                cv2.circle(
                    im, tuple(fp.ips_rc[[1, 0], ip_i] +
                              np.array([self.tiling.x_offsets[fp_i], 0])),
                    5, (0, 255, 0), 1, cv2.LINE_AA)
        return im

    def hicklable(self):
        return [self.factor, [fp.hicklable() for fp in self.forward_passes],
                self.tiling.hicklable()]

    def getPatchesSortedByScore(self, image, radius):
        n = len(self.forward_passes)
        if image is not None:
            pyramid = Pyramid(image, self.factor, n)
            images = pyramid.images
        else:
            images = [fp.image_scores for fp in self.forward_passes]
        batches = np.concatenate([
            self.forward_passes[i].getPatchesSortedByScore(
                images[i], radius) for i in range(n)])
        return batches[self.flatSorting()]


def forwardPassFromHicklable(hicklable):
    return ForwardPass(
        hicklable[0],
        [system.forwardPassFromHicklable(i) for i in hicklable[1]],
        tilingFromHicklable(hicklable[2]))


class ForwardPasser(object):
    def __init__(self, forward_passer, factor, size):
        self._forward_passer = forward_passer
        self._factor = float(factor)
        self._size = size

    def __call__(self, image):
        pyramid = Pyramid(image, self._factor, self._size)
        forward_passes = [self._forward_passer(im) for im in pyramid.images]
        ret = ForwardPass(self._factor, forward_passes, pyramid.tiling)
        ret.selectTop(self._forward_passer.num_ips,
                      self._forward_passer.nms_size)
        return ret


def plotMatches(forward_passes, matched_indices, inliers, name):
    inlier_indices = [matched[inliers] for matched in matched_indices]
    inlier_ips_in_tiled = [forward_passes[i].ipsInTiledImageWithFlatIndices(
        inlier_indices[i]) for i in [0, 1]]
    debug_ims = [fp.debugImage() for fp in forward_passes]
    im = np.zeros([debug_ims[0].shape[0] + debug_ims[1].shape[0],
                   np.maximum(debug_ims[0].shape[1], debug_ims[1].shape[1]), 3])
    y_offset = debug_ims[0].shape[0]
    im[:debug_ims[0].shape[0], :debug_ims[0].shape[1]] = debug_ims[0]
    im[y_offset:y_offset+debug_ims[1].shape[0], :debug_ims[1].shape[1]] = \
        debug_ims[1]
    inl0 = [tuple(inlier_ips_in_tiled[0][[1, 0], i])
            for i in range(inlier_ips_in_tiled[0].shape[1])]
    inl1 = [tuple(inlier_ips_in_tiled[1][[1, 0], i] + np.array([0, y_offset]))
            for i in range(inlier_ips_in_tiled[1].shape[1])]
    assert len(inl0) == len(inl1)
    for i in range(len(inl0)):
        cv2.line(im, inl0[i], inl1[i], (0, 255, 0), 1, cv2.LINE_AA)
    if not os.path.exists(hyperparams.choutRootPath()):
        os.makedirs(hyperparams.choutRootPath())
    filename = os.path.join(hyperparams.choutRootPath(), name + '.png')
    cv2.imwrite(filename, im)
