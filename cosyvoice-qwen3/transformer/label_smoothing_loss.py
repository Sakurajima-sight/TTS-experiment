# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
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
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失。

    在标准的交叉熵损失中，标签的数据分布如下：
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    在平滑版本的交叉熵损失中，一些概率从真实标签的概率（1.0）中分配到其他标签上。

    例如：
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    参数：
        size (int): 类别数量
        padding_idx (int): 填充类别的索引，该类别会在计算损失时被忽略
        smoothing (float): 平滑因子（0.0 表示传统的交叉熵损失）
        normalize_length (bool): 如果为 True，则按序列长度归一化损失；如果为 False，则按批次大小归一化损失
    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """构造一个 LabelSmoothingLoss 对象。

        参数：
            size (int): 类别的数量
            padding_idx (int): 填充类别的索引
            smoothing (float): 平滑因子
            normalize_length (bool): 是否按序列长度归一化损失
        """
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="none")  # 使用KL散度损失
        self.padding_idx = padding_idx  # 填充类别索引
        self.confidence = 1.0 - smoothing  # 置信度
        self.smoothing = smoothing  # 平滑因子
        self.size = size  # 类别数量
        self.normalize_length = normalize_length  # 是否按序列长度归一化

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算模型输出与目标之间的损失。

        模型输出和目标标签的张量被展平成 (batch * seqlen, class) 形状，并应用一个掩码
        来屏蔽填充部分，不计算填充部分的损失。

        参数：
            x (torch.Tensor): 模型预测，形状为 (batch, seqlen, class)
            target (torch.Tensor): 目标标签，形状为 (batch, seqlen)，其中填充部分使用 padding_idx 标记

        返回：
            loss (torch.Tensor): 计算得到的 KL 损失，标量浮动值
        """
        assert x.size(2) == self.size  # 确保预测的类别数与标签数相同
        batch_size = x.size(0)  # 获取批次大小
        x = x.view(-1, self.size)  # 展平预测张量，形状变为 (batch*seqlen, class)
        target = target.view(-1)  # 展平目标标签张量，形状变为 (batch*seqlen)
        
        # 创建一个与 x 形状相同的全零张量，用于存储标签平滑的目标分布
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.size - 1))  # 对非目标类的概率赋值为平滑因子
        
        ignore = target == self.padding_idx  # 找出填充部分，填充部分不计算损失
        total = len(target) - ignore.sum().item()  # 总共有效的标签数（忽略填充部分）
        
        target = target.masked_fill(ignore, 0)  # 将目标标签中的填充部分标记为 0，避免 -1 索引
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # 将目标标签对应的概率设置为 confidence
        
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)  # 计算 KL 散度损失
        
        denom = total if self.normalize_length else batch_size  # 如果需要按序列长度归一化，则使用 total，否则使用 batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom  # 对填充部分不计算损失，并归一化
