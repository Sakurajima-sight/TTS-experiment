import random
import json
import math
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from utils.file_utils import read_lists, read_json_lists


class Processor(IterableDataset):
    """
    处理器类，用于对数据集进行处理
    """
    def __init__(self, source: IterableDataset, f: callable, *args, **kw):
        """
        初始化Processor类
        
        Args:
            source (IterableDataset): 输入的数据源
            f (callable): 处理函数
            *args: 传递给处理函数的额外参数
            **kw: 传递给处理函数的额外关键字参数
        """
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch: int):
        """
        设置数据集的epoch
        
        Args:
            epoch (int): 当前的epoch值
        """
        self.source.set_epoch(epoch)

    def __iter__(self):
        """
        返回一个处理后的数据集迭代器

        Returns:
            Iterator: 返回一个经过处理的数据集迭代器
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f: callable):
        """
        应用新的处理函数，返回新的Processor对象

        Args:
            f (callable): 新的处理函数
        
        Returns:
            Processor: 处理后的Processor对象
        """
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    """
    分布式采样器，用于在多个进程之间划分数据
    """
    def __init__(self, shuffle: bool = True, partition: bool = True):
        """
        初始化分布式采样器
        
        Args:
            shuffle (bool): 是否对数据进行洗牌
            partition (bool): 是否进行数据划分
        """
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self) -> dict:
        """
        更新分布式采样器的信息，包括rank、world_size等
        
        Returns:
            dict: 当前的分布式环境信息
        """
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch: int):
        """
        设置当前epoch
        
        Args:
            epoch (int): 当前的epoch值
        """
        self.epoch = epoch

    def sample(self, data: list) -> list:
        """
        根据rank、world_size和num_workers对数据进行采样
        
        Args:
            data (List): 输入的数据列表
        
        Returns:
            List: 经过采样的数据列表
        """
        data = list(range(len(data)))
        # 强制数据列表为偶数
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            if len(data) < self.world_size:
                data = data * math.ceil(self.world_size / len(data))
                data = data[:self.world_size]
            data = data[self.rank::self.world_size]
        if len(data) < self.num_workers:
            data = data * math.ceil(self.num_workers / len(data))
            data = data[:self.num_workers]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    """
    数据列表类，用于加载数据并进行分布式采样
    """
    def __init__(self, lists: list, shuffle: bool = True, partition: bool = True):
        """
        初始化DataList类
        
        Args:
            lists (list): 数据列表
            shuffle (bool): 是否进行数据洗牌
            partition (bool): 是否进行数据划分
        """
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch: int):
        """
        设置数据集的epoch
        
        Args:
            epoch (int): 当前的epoch值
        """
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        """
        返回一个分布式采样后的数据集迭代器

        Returns:
            Iterator: 返回一个经过分布式采样的数据集迭代器
        """
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        for index in indexes:
            data = dict(src=self.lists[index])
            data.update(sampler_info)
            yield data


def Dataset(data_list_file: str,
            data_pipeline: list,
            mode: str = 'train',
            gan: bool = False,
            shuffle: bool = True,
            partition: bool = True,
            tts_file: str = '',
            prompt_utt2data: str = '') -> IterableDataset:
    """
    构建数据集
    
    本函数中有两个洗牌阶段。第一个是全局洗牌，按shard或原始文件级别进行；第二个是按训练样本级别进行的全局洗牌。
    
    Args:
        data_list_file (str): 数据列表文件路径
        data_pipeline (list): 数据处理管道
        mode (str): 模式 ('train' 或 'inference')
        gan (bool): 是否为GAN模式
        shuffle (bool): 是否对数据进行全局洗牌
        partition (bool): 是否进行数据划分
        tts_file (str): TTS文件路径（仅在'inference'模式下使用）
        prompt_utt2data (str): 提示和数据映射文件路径（仅在'inference'模式下使用）
    
    Returns:
        IterableDataset: 返回构建好的数据集
    """
    assert mode in ['train', 'inference']
    lists = read_lists(data_list_file)
    if mode == 'inference':
        with open(tts_file) as f:
            tts_data = json.load(f)
        utt2lists = read_json_lists(prompt_utt2data)
        # 在推理模式下，过滤掉不必要的文件
        lists = list({utt2lists[utt] for utt in tts_data.keys() if utt2lists[utt] in lists})
    dataset = DataList(lists,
                       shuffle=shuffle,
                       partition=partition)
    if mode == 'inference':
        # 在推理模式下，映射部分参数到parquet_opener函数
        data_pipeline[0] = partial(data_pipeline[0], tts_data=tts_data)
    if gan is True:
        # 在GAN模式下，映射部分参数到padding函数
        data_pipeline[-1] = partial(data_pipeline[-1], gan=gan)
    for func in data_pipeline:
        dataset = Processor(dataset, func, mode=mode)
    return dataset
