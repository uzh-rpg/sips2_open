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
import sys

FLAGS = flags.FLAGS

# Hyperparams
flags.DEFINE_integer('d', 10, 'NN depth')
flags.DEFINE_integer('w', 128, 'Width (channel count)')
flags.DEFINE_bool('res', False, 'Use residual connections?')
flags.DEFINE_string('tds', 'tmbrc', 'Train data set (hp,tmbrc)')
flags.DEFINE_string('pck', 'tr', 'Pair pick method (gt, tr, tt)')
flags.DEFINE_float('ol', 0.3, 'KLT overlap between picked frames')
flags.DEFINE_bool('prob', True, 'Probabilistic interpretation of output')
flags.DEFINE_integer('num_scales', 1, 'How many scales to extract.')
flags.DEFINE_float('scale_factor', 0.71, 'Factor between scales.')
flags.DEFINE_bool('pbs', True, 'Pre-bias sigmoid?')
flags.DEFINE_bool('augment', True, 'Augment?')
flags.DEFINE_float('scale_aug_range', 1., 'Scale augmentation range (2^n)')
flags.DEFINE_bool('lk', True, 'Use leaky ReLU?')

# Training control
flags.DEFINE_integer('its', 20000, 'Iteration count')
flags.DEFINE_integer('val_every', 250, 'Validate every n steps')
flags.DEFINE_bool('dump_out', False, 'Dump final channel as image')
# annealing does not seem to help
flags.DEFINE_integer('lr', 5, 'Negative log of learning rate.')
flags.DEFINE_bool('klti', False, 'Inliers from KLT?')

# Evaluation control
flags.DEFINE_integer('num_test_pts', 500, 'Amount of points extracted')
flags.DEFINE_string('ds', 'kt', 'Evaluation dataset (kt,eu)')
flags.DEFINE_integer('nms', 5, 'Nonmax suppression radius')
flags.DEFINE_bool('testing', False, 'Evaluate testing instead of validation?')
flags.DEFINE_integer('k', 10, 'Required inlier count')

flags.DEFINE_integer('fpbs', 8, 'Forward pass batch size')

flags.DEFINE_string('baseline', '', 'Evaluate baseline?')

flags.DEFINE_bool('debug_plot', False, 'Output debug plots?')

flags.DEFINE_bool('val_best', False, 'use best net for validation?')


sys.argv = flags.FLAGS(sys.argv)
