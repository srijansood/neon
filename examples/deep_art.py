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
import requests
from StringIO import StringIO
import numpy as np
from PIL import Image

from neon.models import Model
from neon import NervanaObject
from neon.transforms import Rectlin
from neon.backends import gen_backend
from neon.data.datasets import Dataset
from neon.util.persist import load_obj
from neon.util.argparser import NeonArgparser
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
    size = 575467849

    # edit filepath below if you have the file elsewhere
    _, filepath = Dataset._valid_path_append('data', '', filename)
    if not os.path.exists(filepath):
        Dataset.fetch_dataset(url, filename, filepath, size)
    trained_vgg = load_obj(filepath)

    param_layers = [l for l in model.layers.layers]
    param_dict_list = trained_vgg['model']['config']['layers']

    for layer, params in zip(param_layers, param_dict_list):
        layer.load_weights(params, load_states=True)


def preprocess(im, MIN_LENGTH):
    """
    Subtracts mean VGG image value
    :return: Raw Image for Display purposes
    :return: Tensor representing image
    """
    im = Image.open(im)
    MEAN_VALUES = np.array([123.68, 116.779, 103.939])

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
    
    # Subtract Mean Pixel values
    np_im = im - MEAN_VALUES

    # Swap Dimensions so that channels is first
    np_im_t = np.empty((np_im.shape[2], np_im.shape[0], np_im.shape[1]))
    np_im_t[:] = np_im.transpose((2, 0, 1))

    return im, be.array(np_im_t, dtype=np.float32)


def content_loss(orig, gen, layer):
    """
    :param orig: Original Image Features
    :param gen: Generated Image Features
    :return: Squared Error loss b/w feature representations
    """

    # feature representation
    orig_feat = orig[layer]
    gen_feat = gen[layer]
    
    loss = be.zeros((1))
    loss[:] = 0.5 * be.sum((gen_feat - orig_feat)**2)
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


def get_feats(model, input_tensor, layer_names, layer_indices):
    """
    Performs fprop and returns feature maps
    """
    model.fprop(input_tensor)
    feats = {k: model.layers.layers[i].outputs for k, i in 
            zip(layer_names, layer_indices)}
    return feats

def main():
    parser = NeonArgparser(__doc__, default_overrides=dict(batch_size=1))
    parser.add_argument("--content",
                        help="Content Image", required=True)
    parser.add_argument("--style",
                        help="Style Image", required=True)
    parser.add_argument("--ratio", default=1e-3, type=float,
                        help="Alpha-Beta ratio for content and style")
    parser.add_argument("--min", default=600, type=int, help="Min Image Length for re-scaling")
    parser.add_argument("--art", default='art_out.png',
                        help="Save painting to named file")
    args = parser.parse_args()
    
    global be 
    be = NervanaObject.be 
    # python deep_art.py --content https://tuebingen.mpg.de/typo3temp/pics/1b4f45ef69.jpg --style https://upload.wikimedia.org/wikipedia/commons/9/94/Starry_Night_Over_the_Rhone.jpg

    if os.path.exists(args.content):
        content_raw, content = preprocess(args.content, args.min)
    elif args.content.startswith("http"):
        r = requests.get(args.content)
        content_raw, content = preprocess(StringIO(r.content), args.min) 

    if os.path.exists(args.style):
        style_raw, style = preprocess(args.style, args.min)
    elif args.style.startswith("http"):
        r = requests.get(args.style)
        style_raw, style  = preprocess(StringIO(r.content), args.min) 
   
    # Build Model
    model = build_vgg()
    load_weights(model)
    model.initialize(content.shape)
    
    # Forward Propagation and Featur Extraction
    content_names = ['conv4_2']
    content_indices = [30]
    content_feats = get_feats(model, content, content_names, content_indices)

    style_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    style_indices = [0, 7, 14, 27, 40]
    style_feats = get_feats(model, style, style_names, style_indices)
    
    input_feats = content_feats.copy()
    input_feats.update(style_feats)
    # input_feats = {'content': content_layers, 'style': style_layers}
     
    # Generating Random Image 
    generated = be.array(np.random.uniform(-128, 128, (3, args.min, args.min)))
    gen_feats = get_feats(model, generated, content_names + style_names, 
            content_indices + style_indices)
    
    closs = content_loss(input_feats, gen_feats, 'conv4_2')
    import pdb; pdb.set_trace();
    print(closs.execute())

if __name__ == '__main__':
    main()
