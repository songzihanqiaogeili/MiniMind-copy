"""预训练 JSONL 数据集，每个非空行应包含一个字符串 text 字段。"""

import json
from array import array
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """只保存行偏移索引，取样时读取一条记录；适用于多进程 DataLoader。"""

    def __init__(self, data_path, tokenizer, max_length: int = 512):
        super().__init__()
        if not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer >= 2 for BOS and EOS")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(tokenizer, name, None) is None:
                raise ValueError(f"tokenizer.{name} must be set")
        self.data_path = Path(data_path).resolve()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets = array("Q")
        # 二进制偏移不受 UTF-8 中文字符长度和 Windows 换行方式影响。
        with self.data_path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("Dataset contains no nonempty JSONL records")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        # 每次独立打开文件，避免不同 worker 共享文件游标。
        with self.data_path.open("rb") as stream:
            stream.seek(self.offsets[index])
            raw = stream.readline()
        try:
            sample = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSON at record {index} in {self.data_path}") from exc
        if not isinstance(sample, dict) or not isinstance(sample.get("text"), str):
            raise ValueError(f"Record {index} must contain a string 'text' field")

        budget = self.max_length - 2
        token_ids = (
            self.tokenizer(
                sample["text"], add_special_tokens=False,
                max_length=budget, truncation=True,
            )["input_ids"][:budget]
            if budget else []
        )
        tokens = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        valid_length = len(tokens)
        input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
        input_ids[:valid_length] = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:valid_length] = 1
        labels = input_ids.clone()
        # 按位置屏蔽 padding，保留真实 EOS，即使 PAD 与 EOS 的 ID 相同。
        labels[valid_length:] = -100
        # 不在数据集里 shift：ForCausalLM.forward 会将预测与标签错开一位。
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
