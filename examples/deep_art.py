#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------


"""
Implementation of Deep Art (A Neural Network of Artistic Style)
    http://arxiv.org/pdf/1508.06576v2.pdf
Combines the content of a photograph with a painting's style. This is done by...

Usage:
"""

"""
The weights in the normalised (19 layer VGG) network are scaled such that the
mean activation of each filter over images and positions is equal to one.
Such re-scaling can always be done without changing the output of a
neural network as long as the non-linearities in the network rectifying linear.


Download Normalized Network:
    https://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel
Normalized pretrained Weights:
    https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl
"""

"""
Extras:
- Retain Color: http://blog.deepart.io/2016/06/04/color-independent-style-transfer/
- Multiple Style Images
"""

