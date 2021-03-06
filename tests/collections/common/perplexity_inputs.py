# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2018-2020 William Falcon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple

import torch

from .pl_utils import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES

Input = namedtuple('Input', ["probs", "logits"])


_only_probs = Input(probs=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES), logits=None)

_only_logits1 = Input(probs=None, logits=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))

_only_logits100 = Input(probs=None, logits=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES) * 200 - 100)

_probs_and_logits = Input(
    probs=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    logits=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES) * 200 - 100,
)

_no_probs_no_logits = Input(probs=None, logits=None)
