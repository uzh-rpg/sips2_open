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

import cv2
import numpy as np


def colored(p_image):
    return cv2.applyColorMap((p_image * 255).astype(np.uint8), cv2.COLORMAP_JET)


def tile(batch, rows, cols):
    if not batch.shape[0] == rows * cols:
        raise Exception('%d %d %d' % (batch.shape[0], rows, cols))
    xdim = batch.shape[2]
    ydim = batch.shape[1]

    im = np.zeros([rows * ydim, cols * xdim, 3])
    for r in range(rows):
        for c in range(cols):
            im_i = colored(batch[r * cols + c, :, :])
            # Add red border to find center:
            im_i[0, :, 2] = 255
            im_i[-1, :, 2] = 255
            im_i[:, 0, 2] = 255
            im_i[:, -1, 2] = 255
            im[r * ydim:(r + 1) * ydim, c * xdim:(c + 1) * xdim, :] = im_i

    return im
