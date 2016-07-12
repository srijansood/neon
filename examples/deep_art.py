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
from PIL import Image
import numpy as np

from neon.models import Model
from neon.transforms import Rectlin
from neon.backends import gen_backend
from neon.data.datasets import Dataset
from neon.util.persist import load_obj
from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, GlorotUniform, Xavier
from neon.layers import Conv, Dropout, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer

be = gen_backend(batch_size=64)

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
    size = 575467849

    # edit filepath below if you have the file elsewhere
    filepath = "/Users/srijan-n/Downloads/VGG_E.p"
    # _, filepath = Dataset._valid_path_append('data', '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)
    trained_vgg = load_obj(filepath)

    param_layers = [l for l in model.layers.layers]
    param_dict_list = trained_vgg['model']['config']['layers']

    for layer, params in zip(param_layers, param_dict_list):
        layer.load_weights(params, load_states=True)


def preprocess(im):
    """
    Subtracts mean VGG image value
    :return: Tensor representing image
    """
    im = Image.open(im)
    MEAN_VALUES = np.array([123.68, 116.779, 103.939])
    MIN_LENGTH = 600

    # Resize so that smallest side is 600
    w, h = im.size
    resize_ratio = (min(w, h) * 1.0) / MIN_LENGTH
    w, h = (int(w/resize_ratio), int(h/resize_ratio))
    im = im.resize((w,h))
    w, h = im.size

    # Crop center portion
    left = w//2 - MIN_LENGTH//2
    up = h//2 - MIN_LENGTH//2
    down = h//2 + MIN_LENGTH//2
    right = w//2 + MIN_LENGTH//2
    im = im.crop((left, up, right, down))

    np_im = im - MEAN_VALUES

    return im, be.array(np_im)


def content_loss(orig, gen, layer):
    """
    :param orig: Original Image
    :param gen: Generated Image
    :return: Squared Error loss b/w feature representations
    """

    # feature representation
    orig_feat = orig[layer]
    gen_feat = gen[layer]

    loss = 0.5 * be.sum((gen_feat - orig_feat)**2)
    return loss

def gram_matrix(tensor):
    """
    Represents feature correlations
    """
    # tensor.take([:], 1, i)
    # tensor.take([:], 2, j)
    # return be.dot(i, j)

def style_loss(orig, gen, layer):
    """
    :param orig: Original Image
    :param gen: Generated Image
    :return: Mean squared dist. b/w Gram matrices of orig, gen image
    """
    # feature representation
    orig_feat = orig[layer]
    gen_feat = gen[layer]

    num_filters = orig.shape[1]
    size_feats = orig.shape[2] * orig.shape[3]

    gram_orig = gram_matrix(orig)
    gram_gen = gram_matrix(gen)

    loss = 1./(4 * num_filters**2 * size_feats**2) * \
           be.sum(gram_gen - gram_orig)

    return loss


def main():
    # content_raw, content = preprocess("/Users/srijan-n/Nervana/tubingen.jpg")
    # style_raw, style = preprocess("/Users/srijan-n/Nervana/starry.jpg")
    #
    import ipdb; ipdb.set_trace()
    parser = NeonArgparser(__doc__)
    parser.add_argument("--content",
                        help="Content Image", required=True)
    parser.add_argument("--style",
                        help="Style Image", required=True)
    parser.add_argument("--ratio", default=1e-3, type=float,
                        help="Alpha-Beta ratio for content and style")
    parser.add_argument("--art", default='art_out.png',
                        help="Save painting to named file")
    args = parser.parse_args()


    model = build_vgg()
    load_weights(model)
    print(model.layers.layers[0].W.get())

    layer_names = ['conv4_2', 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1',
                   'conv5_1']
    layer_indices = [30, 0, 7, 14, 27, 40]
    # content_layers = ['conv4_2']
    # style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    layers = {k: model.layers.layers[i] for k, i in
              zip(layer_names, layer_indices)}

    import pdb; pdb.set_trace()
    gram_matrix(model)

if __name__ == '__main__':
    main()