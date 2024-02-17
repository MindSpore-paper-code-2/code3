# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Custom metrics to process batches with incorrect labels."""
import logging

from mindspore.nn.metrics.metric import EvaluationBase


class MetricWrapper(EvaluationBase):
    """
    Metric class for model with error in metrics computing.
    """
    def __init__(
            self,
            metric_cls: EvaluationBase.__class__,
            eval_type='classification'
    ):
        super().__init__(eval_type)
        self.metric = metric_cls(eval_type)

    def clear(self):
        self.metric.clear()

    def update(self, *inputs):
        y_pred, labels = inputs
        try:
            self.metric.update(y_pred, labels)
        except ValueError:
            logging.error(
                'Skip batch! Error in metrics computing. '
                'y_pred[0] shape %s. y_pred[1] shape %s. Labels shape %s',
                y_pred[0].shape, y_pred[1].shape, labels.shape,
                exc_info=True
            )
            return

    def eval(self):
        return self.metric.eval()
