import logging
import random

import pyarrow.parquet as pq
from io import BytesIO
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pyworld as pw

AUDIO_FORMAT_SETS = {'flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'}


def parquet_opener(data, mode='train', tts_data={}):
    """ 
    给定url或本地文件，返回文件描述符
    原地操作

    参数:
        data (Iterable[str]): url或本地文件列表

    返回:
        Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        url = sample['src']
        try:
            for df in pq.ParquetFile(url).iter_batches(batch_size=64):
                df = df.to_pandas()
                for i in range(len(df)):
                    if mode == 'inference' and df.loc[i, 'utt'] not in tts_data:
                        continue
                    sample.update(dict(df.loc[i]))
                    if mode == 'train':
                        # 注意：不要直接返回sample，必须初始化一个新的字典
                        yield {**sample}
                    else:
                        for index, text in enumerate(tts_data[df.loc[i, 'utt']]):
                            yield {**sample, 'tts_index': index, 'tts_text': text}
        except Exception as ex:
            logging.warning('打开{}失败，异常信息{}'.format(url, ex))


def tokenize(data, get_tokenizer, allowed_special, mode='train'):
    """ 
    解码文本为字符或BPE
    原地操作

    参数:
        data: Iterable[{key, wav, txt, sample_rate}]

    返回:
        Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    tokenizer = get_tokenizer()
    for sample in data:
        assert 'text' in sample
        sample['text_token'] = tokenizer.encode(sample['text'], allowed_special=allowed_special)
        if mode == 'inference':
            sample['tts_text_token'] = tokenizer.encode(sample['tts_text'], allowed_special=allowed_special)
        yield sample


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1,
           mode='train'):
    """ 
    根据特征和标签长度过滤样本
    原地操作

    参数:
        data: Iterable[{key, wav, label, sample_rate}]
        max_length: 丢弃超过max_length(单位：10ms)的语句
        min_length: 丢弃少于min_length(单位：10ms)的语句
        token_max_length: 丢弃超过token_max_length的语句，尤其是在使用字符单位进行英语建模时
        token_min_length: 丢弃少于token_max_length的语句
        min_output_input_ratio: 最小的token_length / feats_length(单位：10ms)比例
        max_output_input_ratio: 最大的token_length / feats_length(单位：10ms)比例

    返回:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))
        sample['speech'] = sample['speech'].mean(dim=0, keepdim=True)
        del sample['audio_data']
        # sample['wav'] 是torch.Tensor，每秒100帧
        num_frames = sample['speech'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['text_token']) < token_min_length:
            continue
        if len(sample['text_token']) > token_max_length:
            continue
        if len(sample['speech_token']) == 0:
            continue
        if num_frames != 0:
            if len(sample['text_token']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['text_token']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=22050, min_sample_rate=16000, mode='train'):
    """ 
    对数据进行重采样。
    原地操作。

    参数:
        data: Iterable[{key, wav, label, sample_rate}]
        resample_rate: 目标重采样率

    返回:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['speech']
        if sample_rate != resample_rate:
            if sample_rate < min_sample_rate:
                continue
            sample['sample_rate'] = resample_rate
            sample['speech'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        max_val = sample['speech'].abs().max()
        if max_val > 1:
            sample['speech'] /= max_val
        yield sample


def compute_fbank(data,
                  feat_extractor,
                  token_mel_ratio=0,
                  mode='train'):
    """ 
    提取fbank特征

    参数:
        data: Iterable[{key, wav, label, sample_rate}]

    返回:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        feat = feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
        if token_mel_ratio != 0:
            # 修剪以对齐speech_token和speech_feat
            token_len = int(min(feat.shape[0] / token_mel_ratio, sample["speech_token"].shape[0]))
            feat = feat[:token_mel_ratio * token_len]
            sample["speech_token"] = sample["speech_token"][:token_len]
        sample['speech_feat'] = feat
        yield sample


def parse_embedding(data, normalize, mode='train'):
    """ 
    解析utt_embedding/spk_embedding

    参数:
        data: Iterable[{key, wav, label, sample_rate}]

    返回:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        sample['utt_embedding'] = torch.tensor(sample['utt_embedding'], dtype=torch.float32)
        sample['spk_embedding'] = torch.tensor(sample['spk_embedding'], dtype=torch.float32)
        if normalize:
            sample['utt_embedding'] = F.normalize(sample['utt_embedding'], dim=0)
            sample['spk_embedding'] = F.normalize(sample['spk_embedding'], dim=0)
        yield sample


def shuffle(data, shuffle_size=10000, mode='train'):
    """ 
    本地打乱数据

    参数:
        data: Iterable[{key, feat, label}]
        shuffle_size: 打乱的缓存大小

    返回:
        Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # 剩余样本
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500, mode='train'):
    """ 
    根据特征长度对数据进行排序。
    排序操作在打乱数据之后、批次操作之前进行，以便将具有相似长度的语句分到同一个批次中，
    `sort_size`应小于`shuffle_size`

    参数:
        data: Iterable[{key, feat, label}]
        sort_size: 排序时的缓存大小

    返回:
        Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['speech_feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # 剩余的样本
    buf.sort(key=lambda x: x['speech_feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ 
    根据`batch_size`对数据进行静态批次划分

    参数:
        data: Iterable[{key, feat, label}]
        batch_size: 批次大小

    返回:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode='train'):
    """ 
    根据批次内的总帧数进行动态批次划分，直到达到`max_frames_in_batch`

    参数:
        data: Iterable[{key, feat, label}]
        max_frames_in_batch: 每个批次内的最大帧数

    返回:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'speech_feat' in sample
        assert isinstance(sample['speech_feat'], torch.Tensor)
        new_sample_frames = sample['speech_feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000, mode='train'):
    """ 
    静态或动态批次的封装器
    """
    if mode == 'inference':
        return static_batch(data, 1)
    else:
        if batch_type == 'static':
            return static_batch(data, batch_size)
        elif batch_type == 'dynamic':
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal('不支持的批次类型 {}'.format(batch_type))


def padding(data, use_spk_embedding, mode='train', gan=False):
    """ 
    对数据进行填充，生成训练数据

    参数:
        data: Iterable[List[{key, feat, label}]]
        
    返回:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        speech_feat_len = torch.tensor([x['speech_feat'].size(1) for x in sample],
                                       dtype=torch.int32)
        order = torch.argsort(speech_feat_len, descending=True)

        utts = [sample[i]['utt'] for i in order]
        speech = [sample[i]['speech'].squeeze(dim=0) for i in order]
        speech_len = torch.tensor([i.size(0) for i in speech], dtype=torch.int32)
        speech = pad_sequence(speech, batch_first=True, padding_value=0)
        speech_token = [torch.tensor(sample[i]['speech_token']) for i in order]
        speech_token_len = torch.tensor([i.size(0) for i in speech_token], dtype=torch.int32)
        speech_token = pad_sequence(speech_token,
                                    batch_first=True,
                                    padding_value=0)
        speech_feat = [sample[i]['speech_feat'] for i in order]
        speech_feat_len = torch.tensor([i.size(0) for i in speech_feat], dtype=torch.int32)
        speech_feat = pad_sequence(speech_feat,
                                   batch_first=True,
                                   padding_value=0)
        text = [sample[i]['text'] for i in order]
        text_token = [torch.tensor(sample[i]['text_token']) for i in order]
        text_token_len = torch.tensor([i.size(0) for i in text_token], dtype=torch.int32)
        text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
        utt_embedding = torch.stack([sample[i]['utt_embedding'] for i in order], dim=0)
        spk_embedding = torch.stack([sample[i]['spk_embedding'] for i in order], dim=0)
        batch = {
            "utts": utts,
            "speech": speech,
            "speech_len": speech_len,
            "speech_token": speech_token,
            "speech_token_len": speech_token_len,
            "speech_feat": speech_feat,
            "speech_feat_len": speech_feat_len,
            "text": text,
            "text_token": text_token,
            "text_token_len": text_token_len,
            "utt_embedding": utt_embedding,
            "spk_embedding": spk_embedding,
        }
        if gan is True:
            # 在GAN训练中，我们需要pitch_feat
            pitch_feat = [sample[i]['pitch_feat'] for i in order]
            pitch_feat_len = torch.tensor([i.size(0) for i in pitch_feat], dtype=torch.int32)
            pitch_feat = pad_sequence(pitch_feat,
                                      batch_first=True,
                                      padding_value=0)
            batch["pitch_feat"] = pitch_feat
            batch["pitch_feat_len"] = pitch_feat_len
        else:
            # 仅在GAN训练中需要speech，为节省内存，删除它
            del batch["speech"]
            del batch["speech_len"]
        if mode == 'inference':
            tts_text = [sample[i]['tts_text'] for i in order]
            tts_index = [sample[i]['tts_index'] for i in order]
            tts_text_token = [torch.tensor(sample[i]['tts_text_token']) for i in order]
            tts_text_token_len = torch.tensor([i.size(0) for i in tts_text_token], dtype=torch.int32)
            tts_text_token = pad_sequence(tts_text_token, batch_first=True, padding_value=-1)
            batch.update({'tts_text': tts_text,
                          'tts_index': tts_index,
                          'tts_text_token': tts_text_token,
                          'tts_text_token_len': tts_text_token_len})
        if use_spk_embedding is True:
            batch["embedding"] = batch["spk_embedding"]
        else:
            batch["embedding"] = batch["utt_embedding"]
        yield batch
