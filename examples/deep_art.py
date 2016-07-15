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
import numpy as np
from PIL import Image
from StringIO import StringIO

from neon.layers.layer import Layer,interpret_in_shape, DataTransform

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

class FakeLayer(Layer):

    def __init__(self, nout, name=None):
        init = Xavier(local=True)
        super(FakeLayer, self).__init__(name, "Disabled")
        self.nout = nout

    def __str__(self):
        return "Fake Layer '%s': %d inputs, %d outputs" % (
            self.name, self.nin, self.nout)

    def configure(self, in_obj):
        super(FakeLayer, self).configure(in_obj)

        # shape of the input is in (# input features, batch_size)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)

        # shape of the output is (# output units, batch_size)
        self.out_shape = (self.nout, self.nsteps)

        # if the shape of the weights have not been allocated,
        # we know that his layer's W is a tensor of shape (# outputs, # inputs).
        self.weight_shape = (self.nout, self.nin)

        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        return self.inputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        return error

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

    # Fake layer to allocated deltas buffer to conv1_1
    layers.append(DataTransform(lambda x: x))

    for nofm, i in zip([64, 128, 256, 512, 512], xrange(1, 6)):
        layers.append(Conv((3, 3, nofm), name="conv{}_1".format(i), **conv_params))
        if i == 5:
            break
        layers.append(Conv((3, 3, nofm), name="conv{}_2".format(i), **conv_params))
        if nofm > 128:
            layers.append(Conv((3, 3, nofm), name="conv{}_3".format(i), **conv_params))
            layers.append(Conv((3, 3, nofm), name="conv{}_4".format(i), **conv_params))
        layers.append(Pooling(2, op="avg", strides=2, name="pool{}".format(i)))

    model = Model(layers=layers)

    return model


def init_model_dict():
    global model_dict 
    model_dict = dict()
    for layer in model.layers.layers:
        model_dict[layer.name] = layer


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
    param_layers = param_layers[1:] # Skip weight loading for DataTransform
    param_dict_list = trained_vgg['model']['config']['layers']

    for layer, params in zip(param_layers, param_dict_list):
            layer.load_weights(params, load_states=True)


def preprocess(im, min_length):
    """
    Subtracts mean VGG image value
    :return: Raw Image for Display purposes
    :return: Tensor representing image
    """
    im = Image.open(im)
    MEAN_VALUES = np.array([123.68, 116.779, 103.939])

    # Resize so that smallest side is 600
    w, h = im.size
    resize_ratio = (min(w, h) * 1.0) / min_length
    w, h = (int(w/resize_ratio), int(h/resize_ratio))
    im = im.resize((w,h))
    w, h = im.size

    # Crop center portion
    left = w//2 - min_length//2
    up = h//2 - min_length//2
    down = h//2 + min_length//2
    right = w//2 + min_length//2
    im = im.crop((left, up, right, down))
    
    # Subtract Mean Pixel values
    np_im = im - MEAN_VALUES
    
    # Normalize values (b/w -1 and +1)
    np_im = np_im / 128.0

    # Swap Dimensions so that channels is first
    np_im_t = np.empty((np_im.shape[2], np_im.shape[0], np_im.shape[1]))
    np_im_t[:] = np_im.transpose((2, 0, 1))

    return im, be.array(np_im_t, dtype=np.float32)


def get_feats(input_tensor, layer_names):
    """
    Performs fprop and returns feature maps
    """
    model.fprop(input_tensor)
    feats = dict()
    for layer in layer_names:
        output = model_dict[layer].outputs
        feats[layer] = be.zeros(output.shape).copy(output)
    return feats


def content_loss(orig, gen, layer):
    """
    :param orig: Original Image Features
    :param gen: Generated Image Features
    :return: Squared Error loss b/w feature representations
    :return: Derivative of Loss w.r.t activations
    """
    loss = 0.5 * be.sum((gen[layer] - orig[layer])**2)
    derivative = (gen[layer] - orig[layer]).asnumpyarray()
    derivative[gen[layer].asnumpyarray() < 0] = 0
    return loss.asnumpyarray()[0][0], derivative


def gram_matrix(vector, layer_shape):
    """
    Represents feature correlations
    """
    tensor = vector.reshape(layer_shape)
    gram_matrix = be.sum(be.dot(tensor, tensor.transpose()))
    return gram_matrix


def style_loss(orig, gen, layer):
    """
    :param orig: Original Image
    :param gen: Generated Image
    :return: Mean squared dist. b/w Gram matrices of orig, gen image
    :return: Derivative w.r.t activations in given layer
    """
    # feature representations
    orig_feat = orig[layer]
    gen_feat = gen[layer]

    layer_shape = model_dict[layer].out_shape
    num_filters = layer_shape[0]
    size_feats = layer_shape[1] * layer_shape[2]
    
    gram_orig = gram_matrix(orig_feat, (num_filters, size_feats))
    gram_gen = gram_matrix(gen_feat, (num_filters, size_feats))

    const = 1.0 / (num_filters**2 * size_feats**2)
    loss = 0.25 * const * be.sum(gram_gen - gram_orig)

    derivative = (const * gen_feat.transpose() * (gram_gen - gram_orig)).\
        asnumpyarray()
    derivative[gen_feat.transpose().asnumpyarray() < 0] = 0

    return loss.asnumpyarray()[0][0], derivative


def bprop(error, layer_list, alpha=1.0, beta=0.0):
    for l in reversed(layer_list):
        if l is layer_list[1]:
            return layer_list[1].deltas
        altered_tensor = l.be.distribute_data(error, l.parallelism)
        if altered_tensor:
            l.revert_list.append(altered_tensor)

        from neon.layers.layer import BranchNode
        if type(l.prev_layer) is BranchNode or l is layer_list[0]:
            error = l.bprop(error, alpha, beta)
        else:
            error = l.bprop(error)

        for tensor in l.revert_list:
            model.layers.be.revert_tensor(tensor)

    return layer_list[1].deltas


def get_layer_list(layer_name):
    out_list = []
    for l in model.layers.layers:
        out_list.append(l)
        if (l.name == layer_name):
            return out_list


def total_loss(content_names, style_names, generated_image):
    """
    Gives Total loss (a*content_loss + b*style_loss)
    Also sets global derivative (for gradient)
    """

    # Forward Propagation and Feature Extraction
    gen_feats = get_feats(generated_image, content_names + style_names)

    c_loss = s_loss = 0
    c_diff = []
    s_diff = []

    c_contrib = 1.0 / len(content_names)
    for layer in reversed(content_names):
        res = content_loss(content_feats, gen_feats, layer)
        c_loss += c_contrib * res[0]
        delta = bprop(be.array(res[1]), get_layer_list(layer))
        c_diff.append(delta)


    s_contrib = 1.0 / len(style_names)
    for layer in reversed(style_names):
        res = style_loss(style_feats, gen_feats, layer)
        s_loss += s_contrib * res[0]
        delta = bprop(be.array(res[1]), get_layer_list(layer))
        s_diff.append(delta)

    loss = alpha * c_loss + beta * s_loss
    content_derivative = c_contrib * sum(c_diff)
    style_derivative = s_contrib * sum(s_diff)
    global total_derivative
    total_derivative = (alpha * content_derivative + beta * style_derivative).\
        asnumpyarray()

    import pdb; pdb.set_trace()
    return loss


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
    global model
    model = build_vgg()
    init_model_dict()
    load_weights(model)
    # fakeLayer = DataTransform(lambda x: x)
    # model.layers._layers = [fakeLayer] + model.layers._layers
    # model.layers.layers = [fakeLayer] + model.layers.layers
    model.initialize(content.shape)

    # Initialize alpha-beta
    global alpha, beta
    alpha = args.ratio * 1.0
    beta = 1.0

    # Layers used for Content and Style Representations
    content_names = ['conv4_2']
    style_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    # Forward Propagation and Feature Extraction
    global content_feats, style_feats
    content_feats = get_feats(content, content_names)
    style_feats = get_feats(style, style_names)

    # Generating Random Image
    # TODO use neon.initializers.Uniform ?
    generated = be.array(np.random.uniform(-1, 1, (3, args.min, args.min)))

    # Calculate total loss
    loss = total_loss(content_names, style_names, generated)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
