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
from typing import List, Tuple, Type

import mindspore
from mindspore import nn
from mindspore import ops as F

from .common import LayerNorm2d


class MaskDecoder(nn.Cell):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Cell,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Cell] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Cell): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Cell): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.SequentialCell(
            nn.Conv2dTranspose(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2, has_bias=True),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2dTranspose(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2, has_bias=True),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.CellList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def construct(
        self,
        image_embeddings: mindspore.Tensor,
        image_pe: mindspore.Tensor,
        sparse_prompt_embeddings: mindspore.Tensor,
        dense_prompt_embeddings: mindspore.Tensor,
        multimask_output: bool,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (mindspore.Tensor): the embeddings from the image encoder
          image_pe (mindspore.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (mindspore.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (mindspore.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          mindspore.Tensor: batched predicted masks
          mindspore.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: mindspore.Tensor,
        image_pe: mindspore.Tensor,
        sparse_prompt_embeddings: mindspore.Tensor,
        dense_prompt_embeddings: mindspore.Tensor,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        # Concatenate output tokens
        output_tokens = F.cat([self.iou_token.embedding_table.value(),\
                               self.mask_tokens.embedding_table.value()], axis=0)
        output_tokens = output_tokens.unsqueeze(0).broadcast_to((sparse_prompt_embeddings.shape[0], -1, -1))
        tokens = F.cat((output_tokens, sparse_prompt_embeddings), axis=1)

        # Expand per-image data in batch direction to be per-mask
        src = F.repeat_interleave(image_embeddings, tokens.shape[0], axis=0)
        src = src + dense_prompt_embeddings
        pos_src = F.repeat_interleave(image_pe, tokens.shape[0], axis=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.swapaxes(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[mindspore.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = F.stack(hyper_in_list, axis=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Cell):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList(
            [nn.Dense(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )
        self.sigmoid_output = sigmoid_output

    def construct(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
