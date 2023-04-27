# coding=utf-8
# Copyright 2020 Lindsay Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for utils.mnist_corrupted_environment."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import time

import numpy as np

from tf_agents.environments import mnist_corrupted_py_environment as mcorr
from tf_agents.specs import array_spec
from tf_agents.utils import test_utils



class MNISTCorruptedPyEnvironmentTest( test_utils.TestCase ):

  def setUp(self):

    self._classifier = None
    super(MNISTCorruptedPyEnvironmentTest, self).__init__()
    self._classifier = mcorr.MnistClassifier_DNN()
    self._classifier.train()

    self._env = mcorr.MnistCorrupted( self._classifier )

  def test_translate_diagonal( self ) :
    """Test the translation actions.  This also tests the reset function if the maximum number of steps is less than 75."""

    # reset and draw the environment
    self._env.reset()
    fig = self._env.draw()
    time.sleep( 3 )
    
    # in a loop, translate the image and redraw
    # pause between each redraw
    for i in range( 75 ) :
      print( "translaction count: ", i )
      self._env.step( [1,1] )
      self._env.draw( fig )
      time.sleep( 1 )



if __name__ == '__main__':
  test_utils.main()
