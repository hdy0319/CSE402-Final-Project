import random
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from datasets import load_dataset


class PretrainingDataset(Dataset):
    """
    프리트레인용 데이터셋(Masked Language Modeling).
    Hugging Face 위키피디아 데이터셋에서 샘플을 뽑아 사용합니다.
    """

    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        max_len: int = 128,
        mlm_probability: float = 0.15,
        use_hf: bool = True,
        hf_config: str = "20220301.en",
        num_samples: int = 50000,
        seed: int = 20231424,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability

        if use_hf:
            # 전체 데이터셋 로드 (Memory-map 방식)
            ds = load_dataset(
                "wikipedia",
                hf_config,
                split="train",
                trust_remote_code=True,
                keep_in_memory=False  # 메모리 맵으로 로드
            )
            total = len(ds)
            rng = random.Random(seed)
            # 전체 인덱스 중에서 무작위 샘플링
            self.indices = rng.sample(range(total), num_samples)
            self.dataset = ds
        else:
            raise ValueError("use_hf=False인 경우는 아직 지원하지 않습니다. 파일 기반 로드는 별도 구현 필요합니다.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        text = self.dataset[real_idx]["text"]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        inputs, labels = self.mask_tokens(input_ids)
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "mlm_labels": labels,
        }

    def mask_tokens(self, inputs: torch.Tensor):
        labels = inputs.clone()
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            inputs.tolist(), already_has_special_tokens=True
        )
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        for i, mask_flag in enumerate(special_tokens_mask):
            if mask_flag == 1:
                probability_matrix[i] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% random
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


class FinetuningDataset(Dataset):
    """
    감정분류(Sequence Classification)용 데이터셋.
    CSV 파일에서 'text', 'label' 컬럼을 읽어서 사용합니다.
    """

    def __init__(self, file_path: str, tokenizer: BertTokenizerFast, max_len: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"].strip()
                label = int(row["label"])
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }