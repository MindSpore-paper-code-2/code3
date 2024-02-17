import mindspore.nn as nn
from mindspore.ops import operations as P


class WithLossCell(nn.Cell):
    def __init__(self, backbone):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                              reduction='mean')
        self.reduce_sum = P.ReduceSum()

    def construct(self, data1, data2, label):
        output = self._backbone(data1, data2)
        my_cross_entropy = self.cross_entropy(output, label)
        return my_cross_entropy
