from functools import lru_cache
import torch
from transformers import AutoTokenizer

class QwenTokenizer():
    """
    QwenTokenizer 用于处理 Qwen 模型的分词任务，支持特殊标记的添加和自定义分词器的初始化。

    参数:
        token_path (str): 分词器模型的路径，支持本地路径或 Hugging Face Hub 上的模型名称。
        skip_special_tokens (bool): 是否在解码时跳过特殊 tokens，默认值为 True。
    """
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
        self.special_tokens = special_tokens
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        # 添加自定义特殊 tokens
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        """
        将输入文本编码为 token ids。

        参数:
            text (str): 要编码的文本。
            **kwargs: 其他参数传递给 tokenizer。

        返回:
            list[int]: 编码后的 token ids 列表。
        """
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        """
        将 token ids 解码为文本。

        参数:
            tokens (list[int]): 要解码的 token ids 列表。

        返回:
            str: 解码后的文本。
        """
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    token_path: str,
    skip_special_tokens: bool
) -> QwenTokenizer:
    """
    获取 QwenTokenizer 实例，支持缓存。

    参数:
        token_path (str): 分词器模型的路径，支持本地路径或 Hugging Face Hub 上的模型名称。
        skip_special_tokens (bool): 是否在解码时跳过特殊 tokens，默认值为 True。

    返回:
        QwenTokenizer: 初始化好的 QwenTokenizer 实例。
    """
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)
