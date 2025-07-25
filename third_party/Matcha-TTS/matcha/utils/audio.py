import torch
from librosa.filters import mel as librosa_mel_fn

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    对输入进行动态范围压缩

    参数:
        x (torch.Tensor): 输入张量
        C (float): 压缩因子，默认1
        clip_val (float): 最小阈值，默认1e-5

    返回:
        torch.Tensor: 动态范围压缩后的张量
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    """
    对幅值谱进行动态范围归一化

    参数:
        magnitudes (torch.Tensor): 幅值谱张量

    返回:
        torch.Tensor: 归一化后的幅值谱
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False
):
    """
    计算输入音频的Mel谱

    参数:
        y (torch.Tensor): 输入音频张量，形状为 (batch, length) 或 (length,)
        n_fft (int): FFT窗口大小
        num_mels (int): Mel滤波器数量
        sampling_rate (int): 采样率
        hop_size (int): 帧移
        win_size (int): 窗口大小
        fmin (float): 最低频率
        fmax (float): 最高频率
        center (bool): 是否在输入信号中心添加填充，默认False

    返回:
        torch.Tensor: Mel谱特征
    """
    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
