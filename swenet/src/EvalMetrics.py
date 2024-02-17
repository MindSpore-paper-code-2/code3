# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np


def kappa(TI, RI):
    m, n = TI.shape

    TN = (RI == 0) & (TI == 0)
    TN = float(np.sum(TN == 1))

    TP = (RI != 0) & (TI != 0)
    TP = float(np.sum(TP == 1))

    FP = (RI == 0) & (TI != 0)
    FP = float(np.sum(FP == 1))

    FN = (RI != 0) & (TI == 0)
    FN = float(np.sum(FN == 1))

    Nc = FN + TP
    Nu = FP + TN
    OE = FP + FN
    PRA = (TP + TN) / (m * n)
    PRE = ((TP + FP) * Nc + (FN + TN) * Nu) / (m * n) ** 2

    KC = (PRA - PRE) / (1 - PRE)
    print('===evaluate====\n'
          'TN:{0} '
          'TP:{1} '
          'FP:{2} '
          'FN:{3} '
          'OE:{4} '
          'KC:{5} '.format(TN, TP, FP, FN, OE, KC))
    return KC
