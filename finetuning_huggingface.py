# finetuning.py

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig, BertModel, get_linear_schedule_with_warmup
from model import TinyBERTConfig, BERTForSequenceClassification
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny-BERT Fine-tuning with Huggingface Pretrained for Sentiment Classification"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Huggingface pretrained model id or local path (e.g. prajjwal1/bert-mini)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
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

    # 1) HF config/tokenizer
    hf_cfg = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    # 2) build custom config
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

    # 3) load dataset from hub
    raw_datasets = load_dataset("glue", "sst2")
    def preprocess_fn(examples):
        tok = tokenizer(
            examples["sentence"],
            max_length=args.max_len,
            padding="max_length",
            truncation=True,
        )
        tok["labels"] = examples["label"]
        return tok

    tokenized = raw_datasets.map(preprocess_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = DataLoader(tokenized["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tokenized["validation"], batch_size=args.batch_size)

    # 4) init custom model
    model = BERTForSequenceClassification(custom_cfg, num_labels=args.num_labels).to(device)

    # 5) load HF pretrained encoder
    hf_model = BertModel.from_pretrained(args.model_name_or_path, config=hf_cfg)
    hf_state = hf_model.state_dict()
    enc_state = {k: v for k, v in hf_state.items() if k.startswith("bert.")}
    model.bert.load_state_dict(enc_state, strict=False)
    print("프리트레인된 encoder 가중치 로드 완료")

    # 6) optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 7) training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = torch.zeros_like(input_ids)
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

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"[Epoch {epoch}] Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = torch.zeros_like(input_ids)
                labels = batch["labels"].to(device)

                logits, _ = model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        acc = sum(p==l for p,l in zip(val_preds,val_labels)) / len(val_labels)
        f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"[Epoch {epoch}] Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.bin"))
            print(f"Best model saved (Acc: {best_acc:.4f})")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.bin"))
    print("Finetuning complete.")

if __name__ == "__main__":
    main()