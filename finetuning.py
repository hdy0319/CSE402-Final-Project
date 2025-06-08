import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertModel,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from model import TinyBERTConfig, BERTForSequenceClassification
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_

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
    
    # Huggingface dataset 옵션
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Huggingface dataset 이름 (예: glue)",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset config 이름 (예: sst2)",
    )
    parser.add_argument(
        "--train_split_name",
        type=str,
        default="train",
        help="학습 split 이름",
    )
    parser.add_argument(
        "--validation_split_name",
        type=str,
        default="validation",
        help="검증 split 이름",
    )

    # 로컬 파일 사용 시
    parser.add_argument("--train_file", type=str, default=None, help="로컬 train 파일 경로 (csv/tsv)")
    parser.add_argument("--val_file", type=str, default=None, help="로컬 validation 파일 경로")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Huggingface config/tokenizer 및 custom TinyBERT 설정
    hf_cfg = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
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
    model = BERTForSequenceClassification(custom_cfg, num_labels=args.num_labels).to(device)

    # 2) pretrained encoder weight 로드
    hf_model = BertModel.from_pretrained(args.model_name_or_path, config=hf_cfg)
    encoder_state = {k.replace("bert.", ""): v for k, v in hf_model.state_dict().items()}
    model.bert.load_state_dict(encoder_state, strict=False)
    print("프리트레인된 encoder 가중치 로드 완료")

    # 3) 데이터셋 로드 및 전처리
    if args.dataset_name:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        def preprocess(example):
            text_field = example.get("sentence", example.get("text"))
            tokens = tokenizer(
                text_field,
                truncation=True,
                max_length=args.max_len,
                return_token_type_ids=True,
            )
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "token_type_ids": tokens.get("token_type_ids"),
                "labels": example.get("label", example.get("labels")),
            }
        train_ds = raw_datasets[args.train_split_name].map(preprocess, batched=False)
        val_ds = raw_datasets[args.validation_split_name].map(preprocess, batched=False)
        train_ds.set_format(type="torch", columns=["input_ids","attention_mask","token_type_ids","labels"])
        val_ds.set_format(type="torch", columns=["input_ids","attention_mask","token_type_ids","labels"])

        data_collator = DataCollatorWithPadding(tokenizer)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
    else:
        from data import FinetuningDataset
        train_ds = FinetuningDataset(args.train_file, tokenizer, max_len=args.max_len)
        val_ds = FinetuningDataset(args.val_file, tokenizer, max_len=args.max_len)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 4) optimizer & scheduler 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 5) 학습 및 검증 루프
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
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

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"새로운 최고 성능 모델 저장: {save_path} (Val Acc: {best_val_acc:.4f})")

    final_save = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_save)
    print(f"파인튜닝 완료, 최종 모델 저장: {final_save}")

if __name__ == "__main__":
    main()