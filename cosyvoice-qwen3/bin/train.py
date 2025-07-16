# 命令行： PYTHONPATH=. python bin/train.py
from argparse import Namespace
from functools import partial

from utils.train_utils import init_dataset_and_dataloader
from dataset.processor import *
from tokenizer.tokenizer import get_qwen_tokenizer
import sys
sys.path.append("/workspace/third_party/Matcha-TTS")
from matcha.utils.audio import mel_spectrogram

gan = False

configs = {
    'data_pipeline': [
        parquet_opener,
        partial(
            tokenize, 
            get_tokenizer=partial(
                get_qwen_tokenizer,
                token_path='Qwen/Qwen3-4B',
                skip_special_tokens=True
            ), 
            allowed_special="all"
        ),
        partial(filter, max_length=40960, min_length=100, token_max_length=200, token_min_length=1),
        partial(resample, resample_rate=24000),
        partial(
            compute_fbank,
            feat_extractor=partial(
                mel_spectrogram,
                n_fft=1920,
                num_mels=80,
                sampling_rate=24000,
                hop_size=480,
                win_size=1920,
                fmin=0,
                fmax=8000,
                center=False
            ),
            token_mel_ratio=2
        ),
        partial(parse_embedding, normalize=True),
        partial(shuffle, shuffle_size=1000),
        partial(sort, sort_size=500),  # 新增sort步骤
        partial(batch, batch_type='dynamic', max_frames_in_batch=2000),
        partial(padding, use_spk_embedding=False)
    ]
}

args = Namespace(
    train_data='/workspace/CMU-data/train.data.list',
    cv_data='/workspace/CMU-data/dev.data.list',
    pin_memory=True,
    num_workers=4,
    prefetch=100
)

train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
    init_dataset_and_dataloader(args, configs, gan)
