from torch.utils.data import DataLoader

from dataset.dataset import Dataset

def init_dataset_and_dataloader(args, configs, gan):
    """ 
    初始化数据集和数据加载器（dataloader）

    参数:
        args: argparse.Namespace 对象，包含训练配置如 train_data、cv_data、pin_memory、num_workers、prefetch 等。
        configs: 配置字典，包含 'data_pipeline' 和 'data_pipeline_gan' 等数据处理流程。
        gan (bool): 是否使用 GAN 数据处理流程，True 使用 'data_pipeline_gan'，否则使用 'data_pipeline'。

    返回:
        train_dataset: 训练集 Dataset 对象
        cv_dataset: 验证集 Dataset 对象
        train_data_loader: 训练集 DataLoader
        cv_data_loader: 验证集 DataLoader

    用途：
        初始化训练和验证数据集及其对应的 DataLoader，支持根据 GAN 模式切换数据处理流程。
    """
    data_pipeline = configs['data_pipeline_gan'] if gan is True else configs['data_pipeline']
    train_dataset = Dataset(args.train_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=True, partition=True)
    cv_dataset = Dataset(args.cv_data, data_pipeline=data_pipeline, mode='train', gan=gan, shuffle=False, partition=False)

    # 不推荐使用 persistent_workers=True，因为 Whisper 分词器会在每次循环时打开文件，这可能会导致错误
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader
