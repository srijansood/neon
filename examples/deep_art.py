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
Implementation of Deep Art (A Neural Network of Artistic Style):
    http://arxiv.org/pdf/1508.06576v2.pdf
Model based on VGG:
    https://arxiv.org/pdf/1409.1556.pdf
Combines the content of a photograph with a painting's style. This is done by...

Usage:
"""
import os

from neon.models import Model
from neon.transforms import Rectlin
from neon.data.datasets import Dataset
from neon.util.persist import load_obj
from neon.initializers import Constant, GlorotUniform, Xavier
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer

"""
The weights in the normalised (19 layer VGG) network are scaled such that the
mean activation of each filter over images and positions is equal to one.
Such re-scaling can always be done without changing the output of a
neural network as long as the non-linearities in the network rectifying linear.


Download Normalized Network:
    https://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel
Normalized pretrained Weights:
    https://s3.amazonaws.com/lasagne/recipes/ /imagenet/vgg19_normalized.pkl
"""

"""
Extras:
- Retain Color: http://blog.deepart.io/2016/06/04/color-independent-style-transfer/
- Multiple Style Images
"""

def build_vgg():
    """
    Builds VGG Network (E) based on https://arxiv.org/pdf/1409.1556.pdf
    Uses Average Pooling instead of Max
    """
    conv_params = {'init': Xavier(local=True),
                   'strides': 1,
                   'padding': 1,
                   'bias': Constant(0),
                   'activation': Rectlin()}
    layers = []
    for nofm, i in zip([64, 128, 256, 512, 512], xrange(1, 6)):
        layers.append(Conv((3, 3, nofm), name="conv{}_1".format(i), **conv_params))
        layers.append(Conv((3, 3, nofm), name="conv{}_2".format(i), **conv_params))
        if nofm > 128:
            layers.append(Conv((3, 3, nofm), name="conv{}_3".format(i), **conv_params))
            layers.append(Conv((3, 3, nofm), name="conv{}_4".format(i), **conv_params))
        layers.append(Pooling(2, op="avg", strides=2, name="pool{}".format(i)))

    model = Model(layers=layers)

    return model


def load_weights(model):
    # location and size of the VGG weights file
    url = 'https://s3-us-west-1.amazonaws.com/nervana-modelzoo/VGG/'
    filename = 'VGG_E.p'
    size = 554227541

    # edit filepath below if you have the file elsewhere
    filepath = "/Users/srijan-n/Downloads/VGG_E.p"
    # _, filepath = Dataset._valid_path_append('data', '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)
    trained_vgg = load_obj(filepath)

    param_layers = [l for l in model.layers.layers]
    param_dict_list = trained_vgg['model']['config']['layers']

    for layer, params in zip(param_layers, param_dict_list):
        print(layer.name + ", " + params['config']['name'])
        layer.load_weights(params, load_states=True)


def main():
    model = build_vgg()
    load_weights(model)

if __name__ == '__main__':
    main()