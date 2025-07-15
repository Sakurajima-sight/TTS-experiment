# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
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

import torch

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """创建包含填充部分索引的掩码张量。

    请参见 make_non_pad_mask 的描述。

    参数:
        lengths (torch.Tensor): 每个序列的长度批次 (B,)

    返回:
        torch.Tensor: 包含填充部分索引的掩码张量

    示例:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()  # 如果未指定最大长度，取批次中最大长度
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)  # 创建一个从 0 到 max_len-1 的序列
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)  # 扩展为 (B, max_len) 的矩阵
    seq_length_expand = lengths.unsqueeze(-1)  # 扩展为 (B, 1)
    mask = seq_range_expand >= seq_length_expand  # 生成掩码，长度大于等于当前索引的位置为 True
    return mask