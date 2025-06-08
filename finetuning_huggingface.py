# finetuning.py

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig, BertModel, get_linear_schedule_with_warmup
from model import TinyBERTConfig, BERTForSequenceClassification
from data import FinetuningDataset
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-BERT Fine-tuning with Huggingface Pretrained for Sentiment Classification")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Huggingface pretrained model id or local path (e.g. prajjwal1/bert-mini)",
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
        help="감정분류 학습용 CSV 파일 (columns: text, label)",
    )
    parser.add_argument(
        "--val_file", type=str, required=True,
        help="감정분류 검증용 CSV 파일 (columns: text, label)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="파인튜닝된 모델 저장 디렉토리",
    )
    parser.add_argument("--epochs", type=int, default=3, help="에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=3e-5, help="러닝률")
    parser.add_argument("--max_len", type=int, default=128, help="최대 토큰 길이")
    parser.add_argument("--num_labels", type=int, default=2, help="레이블 수")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Huggingface config/tokenizer 로드
    hf_cfg = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    # 2) 커스텀 TinyBERTConfig 생성 (모양이 동일해야 함)
    custom_cfg = TinyBERTConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        intermediate_size=hf_cfg.intermediate_size,
        max_position_embeddings=hf_cfg.max_position_embeddings,
        hidden_dropout_prob=hf_cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=hf_cfg.attention_probs_dropout_prob,
    )

    # 1) 허깅페이스 허브에서 SST-2(혹은 원하시는 데이터셋) 불러오기
    raw_datasets = load_dataset("glue", "sst2")

    # 2) 토크나이저 로드
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    # 3) 토큰화 함수
    def preprocess_fn(examples):
        tokenized = tokenizer(
            examples["sentence"],              # SST-2 필드명
            max_length=args.max_len,
            padding="max_length",
            truncation=True,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    # 4) 맵 & 포맷
    tokenized = raw_datasets.map(preprocess_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 5) DataLoader
    train_loader = DataLoader(tokenized["train"], batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(tokenized["validation"], batch_size=args.batch_size, shuffle=False)

    # 4) 커스텀 모델 초기화
    model = BERTForSequenceClassification(custom_cfg, num_labels=args.num_labels).to(device)

    # 5) Huggingface pretrained encoder weight 로드
    hf_model = BertModel.from_pretrained(args.model_name_or_path, config=hf_cfg)
    hf_state = hf_model.state_dict()
    encoder_state = {k: v for k, v in hf_state.items() if k.startswith("bert.")}
    model.bert.load_state_dict(encoder_state, strict=False)
    print("프리트레인된 encoder 가중치 로드 완료")

    # 6) 옵티마이저 & 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 7) 파인튜닝 루프
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss = model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"[Epoch {epoch}] Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

        # 검증 단계
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"[Epoch {epoch}] Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # 최고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.bin")
            torch.save(model.state_dict(), save_path)
            print(f"새로운 최고 성능 모델 저장: {save_path} (Val Acc: {best_val_acc:.4f})")

    # 최종 모델 저장
    final_save = os.path.join(args.output_dir, "final_model.bin")
    torch.save(model.state_dict(), final_save)
    print(f"파인튜닝 완료, 최종 모델 저장: {final_save}")


if __name__ == "__main__":
    main()