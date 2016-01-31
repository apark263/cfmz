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
from neon.util.argparser import NeonArgparser
from neon.initializers import Kaiming, IdentityInit
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, ResidualModule, Activation
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import ImageLoader
from neon.callbacks.callbacks import Callbacks

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
parser.add_argument('--depth', type=int, default=9,
                    help='depth of each stage (network depth will be 6n+2)')
args = parser.parse_args()

# setup data provider
imgset_options = dict(inner_size=32, scale_range=40, repo_dir=args.data_dir)
train = ImageLoader(set_name='train', shuffle=True, do_transforms=True, **imgset_options)
test = ImageLoader(set_name='validation', shuffle=False, do_transforms=False, **imgset_options)


def conv_params(fsize, nfm, stride=1, relu=True):
    return dict(fshape=(fsize, fsize, nfm), strides=stride, padding=(1 if fsize > 1 else 0),
                activation=(Rectlin() if relu else None),
                init=Kaiming(local=True),
                batch_norm=True)


def module_factory(nfm, stride=1):
    projection = None if stride == 1 else IdentityInit()
    module = [Conv(**conv_params(3, nfm, stride=stride)),
              Conv(**conv_params(3, nfm, relu=False))]
    return [ResidualModule(module, projection),
            Activation(Rectlin())]


# Structure of the deep residual part of the network:
# args.depth modules of 2 convolutional layers each at feature map depths of 16, 32, 64
nfms = [2**(stage + 4) for stage in sorted(range(3) * args.depth)]
strides = [1] + [1 if cur == prev else 2 for cur, prev in zip(nfms[1:], nfms[:-1])]

# Now construct the network
layers = [Conv(**conv_params(3, 16))]
for nfm, stride in zip(nfms, strides):
    layers.append(module_factory(nfm, stride))
layers.append(Pooling(8, op='avg'))
layers.append(Affine(nout=10, init=Kaiming(local=False), batch_norm=True, activation=Softmax()))

model = Model(layers=layers)
opt = GradientDescentMomentum(0.1, 0.9, wdecay=0.0001, schedule=Schedule([90, 135], 0.1))

# configure callbacks
callbacks = Callbacks(model, eval_set=test, metric=Misclassification(), **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
