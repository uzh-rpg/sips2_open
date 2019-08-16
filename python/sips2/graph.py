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
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


def centerAndExpand(x):
    centered = x - 127
    return tf.expand_dims(centered, 1)


def net(image_batch_nchw, do_pad=False):
    with tf.variable_scope('sips2', reuse=tf.AUTO_REUSE):
        k = [3, 3]
        s = [1, 1]

        w = [FLAGS.w / 2 for _ in range(FLAGS.d / 2)] + \
            [FLAGS.w for _ in range(FLAGS.d / 2)]
        w[-1] = 1

        x = [image_batch_nchw]

        for i in range(FLAGS.d):
            # Pre-activation, for resnet to be valid, and to allow negative
            # values in the final layer.
            if i > 0:
                if FLAGS.lk:
                    x.append(tf.nn.leaky_relu(x[-1]))
                else:
                    x.append(tf.nn.relu(x[-1]))
            else:
                x.append(x[-1])
            if do_pad:
                padding = 'same'
            else:
                padding = 'valid'
            x[-1] = tf.layers.conv2d(
                x[-1], w[i], k, s, padding=padding,
                data_format='channels_first')
            # Add residual connections where valid..
            if FLAGS.res and (i + 1) % 2 == 0 and \
                    x[-3].shape[3] == x[-1].shape[3]:
                x[-1] = x[-1] + x[-3][:, 2:-2, 2:-2, :]

        if FLAGS.prob:
            if FLAGS.pbs:
                return tf.nn.sigmoid(x[-1] - 1.5)
            else:
                return tf.nn.sigmoid(x[-1])
        else:
            return x[-1]


def singleScaleNet(image_batch):
    """ batch of grayscale images [?, ?, ?] to batch of unpadded scores """
    return tf.squeeze(net(centerAndExpand(image_batch)), axis=1)


def multiScaleNet(image_batch):
    """ batch of grayscale images [?, ?, ?] to batch of unpadded scores """
    raise Exception('Deprecated!')
    centered = centerAndExpand(image_batch)

    kernel = tf.constant(.25 * np.ones([2, 2, 1, 1], dtype=np.float32))
    downscaled = tf.nn.conv2d(centered, kernel, [1, 2, 2, 1], 'SAME')

    unscaled_out = net(centered, do_pad=True)
    scaled_out = net(downscaled, do_pad=True)
    if FLAGS.full_ups:
        kernel = tf.constant(np.ones([2, 2, 1, 1], dtype=np.float32))
    upsampled = tf.nn.conv2d_transpose(
        scaled_out, kernel, output_shape=tf.shape(centered),
        strides=[1, 2, 2, 1], padding='SAME')
    combined = tf.squeeze(unscaled_out + upsampled, axis=3)
    # Need to de-pad (hack):
    return combined[:, FLAGS.d:-FLAGS.d, FLAGS.d:-FLAGS.d]


def hinge(x):
    return tf.maximum(0., x)


def inlierLoss(x):
    return -tf.log(x)


def outlierLoss(x):
    return -tf.log(1 - x)


class Graph(object):
    def __init__(self):
        tf.reset_default_graph()
        
        # Feedforward: batch, imdims
        self.ff_input = tf.placeholder(tf.float32, [None, None, None])
        raw_out = singleScaleNet(self.ff_input)
        pad = tf.constant([[0, 0], [FLAGS.d, FLAGS.d], [FLAGS.d, FLAGS.d]])
        self.ff_output = tf.pad(raw_out, pad)

        # Train
        patch_size = 2 * FLAGS.d + 1
        self.train_input = tf.placeholder(
                tf.float32, [None, patch_size, patch_size])
        # should be [?], score value of interest point
        self.train_output = tf.squeeze(singleScaleNet(self.train_input))
        
        # Losses
        self.inlier_mask = tf.placeholder(tf.bool, [None])
        outlier_mask = tf.logical_not(self.inlier_mask)
        inlier_outs = tf.boolean_mask(self.train_output, self.inlier_mask)
        outlier_outs = tf.boolean_mask(self.train_output, outlier_mask)

        if FLAGS.prob:
            self.inlier_loss = tf.reduce_sum(inlierLoss(inlier_outs))
            self.outlier_loss = tf.reduce_sum(outlierLoss(outlier_outs))
        else:
            # Inliers above 1
            self.inlier_loss = tf.reduce_mean(hinge(1. - inlier_outs))
            # Outliers below 0
            self.outlier_loss = tf.reduce_mean(hinge(outlier_outs))
        
        self.loss = self.inlier_loss + self.outlier_loss
        self.train_step = tf.train.AdamOptimizer(10 ** (-FLAGS.lr)).minimize(
            self.loss)
