# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
import queue
import random
from typing import List

import numpy as np
import torch

IGNORE_ID = -1


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor, ignore_label: int) -> torch.Tensor:
    """计算准确率。

    参数:
        pad_outputs (Tensor): 预测张量 (B * Lmax, D)。
        pad_targets (LongTensor): 目标标签张量 (B, Lmax)。
        ignore_label (int): 要忽略的标签 id。

    返回:
        torch.Tensor: 准确率值 (0.0 - 1.0)。
    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1), pad_outputs.size(1)).argmax(2)  # 获取预测结果
    mask = pad_targets != ignore_label  # 创建掩码，忽略指定的标签
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))  # 计算正确的预测数
    denominator = torch.sum(mask)  # 计算有效标签的数量
    return (numerator / denominator).detach()  # 返回准确率
