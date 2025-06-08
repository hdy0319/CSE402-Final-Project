import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_

from model import TinyBERTConfig, BERTForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny-BERT Fine-tuning with custom pretrained TinyBERT"
    )
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default="prajjwal1/bert-mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_split_name", type=str, default="train")
    parser.add_argument("--validation_split_name", type=str, default="validation")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="전체 모델 이어서 학습")
    parser.add_argument("--resume_encoder_only", action="store_true", help="encoder만 불러와서 전이학습")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name_or_path)
    custom_cfg = TinyBERTConfig(vocab_size=tokenizer.vocab_size)
    model = BERTForSequenceClassification(custom_cfg, num_labels=args.num_labels).to(device)

    checkpoint = torch.load(args.pretrain_ckpt, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    enc_state = {k.replace("bert.", ""): v for k, v in state_dict.items() if k.startswith("bert.")}
    model.bert.load_state_dict(enc_state, strict=False)
    print("✅ pretrain된 encoder 가중치 로드 완료")

    if args.dataset_name:
        print("Model vocab size:", model.config.vocab_size)
        print("Embedding weight shape:", model.bert.embeddings.word_embeddings.weight.shape)
        raw_ds = load_dataset(args.dataset_name, args.dataset_config_name)
        def preprocess(ex):
            txt = ex.get("text") or ex.get("sentence") or ""
            tok = tokenizer(
                txt,
                truncation=True,
                max_length=args.max_len,
                return_token_type_ids=True,
                padding="max_length",
            )
            tok["labels"] = ex.get("label", ex.get("labels"))
            return tok

        train_ds = raw_ds[args.train_split_name].map(preprocess, batched=False)
        val_ds = raw_ds[args.validation_split_name].map(preprocess, batched=False)
        for ds in (train_ds, val_ds):
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

        collator = DataCollatorWithPadding(tokenizer)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    else:
        from data import FinetuningDataset
        train_ds = FinetuningDataset(args.train_file, tokenizer, max_len=args.max_len)
        val_ds = FinetuningDataset(args.val_file, tokenizer, max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    start_epoch = 1
    best_acc = 0.0
    ckpt_path = os.path.join(args.output_dir, "best_model.bin")
    if args.resume and os.path.exists(ckpt_path):
        ckpt2 = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt2["model_state_dict"])
        optimizer.load_state_dict(ckpt2["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt2["scheduler_state_dict"])
        start_epoch = ckpt2["epoch"] + 1
        best_acc = ckpt2["best_acc"]
        print(f"✅ epoch {start_epoch}부터 이어서 학습 (best_acc: {best_acc:.4f})")
    elif args.resume_encoder_only and os.path.exists(ckpt_path):
        ckpt2 = torch.load(ckpt_path, map_location=device)
        print("🔁 encoder만 로드하여 전이학습 시작")
        enc_state = {
            k.replace("bert.", ""): v
            for k, v in ckpt2["model_state_dict"].items()
            if k.startswith("bert.")
        }
        model.bert.load_state_dict(enc_state, strict=False)
        start_epoch = 1
        best_acc = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss, corr, tot = 0.0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            token_type_ids = token_type_ids.to(device) if token_type_ids is not None else torch.zeros_like(input_ids)
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
            preds = logits.argmax(dim=-1)
            corr += (preds == labels).sum().item()
            tot += labels.size(0)

        print(f"[Epoch {epoch}] Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {corr/tot:.4f}")

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                token_type_ids = token_type_ids.to(device) if token_type_ids is not None else torch.zeros_like(input_ids)
                labels = batch["labels"].to(device)

                logits, _ = model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                )
                preds = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
        f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"[Epoch {epoch}] Val Acc: {acc:.4f}, Val F1: {f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
            }, ckpt_path)
            print(f"✨ Best model saved (Acc: {best_acc:.4f})")

    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.bin"))
    print("파인튜닝 완료, 최종 모델 저장됨")


if __name__ == "__main__":
    main()