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
from neon.util.persist import load_obj
from neon.transforms import Misclassification
from neon.models import Model
from neon.data import ImageLoader

# parse the command line arguments (generates the backend)
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# setup data provider
test_set = ImageLoader(set_name='validation', repo_dir=args.data_dir,
                       inner_size=32, scale_range=40, do_transforms=False)

model = Model(load_obj(args.model_file), test_set)
print 'Error = %.1f%%' % (model.eval(test_set, metric=Misclassification())*100)
