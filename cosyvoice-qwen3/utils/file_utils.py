import os
import json
import torch
import torchaudio
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

def read_lists(list_file: str) -> list:
    """ 
    读取一个文本文件，每行内容存储到列表中。
    
    参数:
        list_file (str): 文件路径，包含需要读取的文件列表，每行一个文件路径。
        
    返回:
        lists (list): 文件中的每一行作为列表元素，返回列表。
    """
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file: str) -> dict:
    """ 
    读取包含 JSON 文件路径的列表文件，逐个加载 JSON 文件并合并结果。
    
    参数:
        list_file (str): 文件路径，包含需要读取的 JSON 文件路径列表，每行一个文件路径。
        
    返回:
        results (dict): 合并后的 JSON 数据。
    """
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results