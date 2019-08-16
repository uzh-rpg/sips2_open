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
import IPython
import math
import numpy as np
import os

import rpg_common_py.geometry

import hyperparams
import multiscale
import p3p
import sequences
import system

FLAGS = absl.flags.FLAGS


def poseError(pair, T_1_0):
    if T_1_0 is None:
        return None, None

    T_1_0_true = pair.T_0_1.inverse()

    R_err = math.degrees(rpg_common_py.geometry.geodesicDistanceSO3(
        T_1_0.R, T_1_0_true.R))
    t_err = np.linalg.norm(T_1_0.t - T_1_0_true.t)

    return R_err, t_err


def p3pMaskAndError(pair, matched_fps):
    mask, T_1_0 = p3p.ransac(pair, matched_fps, get_pose=True)
    R_err, t_err = poseError(pair, T_1_0)
    return mask, R_err, t_err


def renderMatching(pair, full_fps, matched_fps, inliers):
    renderings = [full_fps[i].render(fallback_im=pair.im[i]) for i in [0, 1]]
    ims = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in pair.im]
    roffsets = [i * ims[0].shape[0] for i in [1, 2]]

    full_im = np.concatenate(
        [ims[0], renderings[0], renderings[1], ims[1]], axis=0)

    matched_ips = [matched_fp.ips_rc for matched_fp in matched_fps]

    inl_ips = [i[:, inliers] for i in matched_ips]
    endpoints = [inl_ips[i] + np.array([[roffsets[i], 0]]).T for i in [0, 1]]
    for i in range(endpoints[0].shape[1]):
        cv2.line(full_im, tuple(endpoints[0][[1, 0], i]),
                 tuple(endpoints[1][[1, 0], i]), (0, 255, 0), 1, cv2.LINE_AA)

    if False:
        outl_ips = [i[:, np.logical_not(inliers)] for i in matched_ips]
        endpoints = [outl_ips[i] + np.array([[roffsets[i], 0]]).T for i in [0, 1]]
        for i in range(endpoints[0].shape[1]):
            cv2.line(full_im, tuple(endpoints[0][[1, 0], i]),
                     tuple(endpoints[1][[1, 0], i]), (0, 0, 255), 1, cv2.LINE_AA)

    outdir = os.path.join(
        'results', 'match_render', hyperparams.methodEvalString())
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, pair.name() + '.png')
    cv2.imwrite(outfile, full_im)


def leastNumForKInliers(pair, forward_passes, k, nmin=0, nmax=500,
                        save_final=False, stereo_cache=None, get_rt=False,
                        get_full=False, verbose=False):
    if nmin == nmax:
        if save_final:
            if FLAGS.num_scales > 1:
                reduced = [i.reducedTo(nmin) for i in forward_passes]
                flat = [i.toFlatForwardPass() for i in reduced]
                matched_indices = system.match([i[0] for i in flat])
                inliers = system.getInliers(
                    pair, flat, matched_indices, stereo_cache=stereo_cache)
                multiscale.plotMatches(
                    reduced, matched_indices, inliers, pair.name())
            else:
                reduced = [i.reducedTo(nmin) for i in forward_passes]
                matched_indices = system.match(reduced)
                inliers = system.getInliers(pair, reduced, matched_indices)
                matched_fps = [reduced[i][matched_indices[i]] for i in [0, 1]]
                renderMatching(pair, reduced, matched_fps, inliers)
                print('%s has %d inliers with %d points' %
                      (pair.name(), np.count_nonzero(inliers), nmin))
        if get_rt:
            assert type(pair) == sequences.PairWithStereo
            if type(forward_passes[0]) == multiscale.ForwardPass:
                forward_passes = [i.toFlatForwardPass() for i in forward_passes]
            reduced = [i.reducedTo(nmin) for i in forward_passes]
            matched_indices = system.match(reduced)
            matched_fps = [forward_passes[i][matched_indices[i]] for i in [0, 1]]
            _, R_err, t_err = p3pMaskAndError(pair, matched_fps)
            return nmin, R_err, t_err
        if get_full:
            assert type(pair) == sequences.PairWithStereo
            if type(forward_passes[0]) == multiscale.ForwardPass:
                forward_passes = [i.toFlatForwardPass() for i in forward_passes]
            reduced = [i.reducedTo(nmin) for i in forward_passes]
            matched_indices = system.match(reduced)
            matched_fps = [forward_passes[i][matched_indices[i]] for i in [0, 1]]
            inlier_mask = p3p.ransac(pair, matched_fps)
            return matched_fps, inlier_mask
        return nmin
    n = (nmin + nmax) / 2
    if type(forward_passes[0]) == multiscale.ForwardPass:
        forward_passes = [i.toFlatForwardPass() for i in forward_passes]
    reduced = [i.reducedTo(n) for i in forward_passes]
    matched_indices = system.match(reduced)
    assert np,max(matched_indices[0]) < len(reduced[0])
    assert np, max(matched_indices[1]) < len(reduced[1])
    inliers = system.getInliers(
        pair, reduced, matched_indices, stereo_cache=stereo_cache)
    if verbose:
        print('%d inliers with n=%d' % (np.count_nonzero(inliers), n))
        print('FP lengths are %d %d' % (
            len(forward_passes[0]), len(forward_passes[1])))
    if np.count_nonzero(inliers) < k:
        return leastNumForKInliers(
            pair, forward_passes, k, nmin=n+1, nmax=nmax, save_final=save_final,
            stereo_cache=stereo_cache, get_rt=get_rt, get_full=get_full,
            verbose=verbose)
    else:
        return leastNumForKInliers(
            pair, forward_passes, k, nmin=nmin, nmax=n, save_final=save_final,
            stereo_cache=stereo_cache, get_rt=get_rt, get_full=get_full,
            verbose=verbose)


def lessThanCurve(values):
    return np.sort(values), \
           np.arange(1, len(values) + 1, dtype=float) / len(values)


def auc(x, y, xmax):
    dy = np.hstack((y[0], y[1:] - y[:-1]))
    filt = x < xmax
    return np.sum((xmax - x[filt]).astype(float) / xmax * dy[filt])


def lessThanAuc(values, xmax):
    x, y = lessThanCurve(values)
    return auc(x, y, xmax)


def succinctness(pairs, forward_passer, k):
    fp_cache = system.ForwardPassCache(forward_passer)
    # if type(pairs[0]) == rpg_datasets_py.hpatches.HPatchPair:
    stats = []
    for pair in pairs:
        print(pair.name())
        stats.append(leastNumForKInliers(
            pair, [fp_cache[im] for im in pair.im], k))
    return lessThanAuc(stats, 200)
