# finetuning.py

import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from model import TinyBERTConfig, BERTForSequenceClassification
from data import FinetuningDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-BERT Fine-tuning for Sentiment Classification")
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        required=True,
        help="pretrain.py로 학습된 모델이 저장된 디렉터리 (pytorch_model.bin 포함)",
    )
    parser.add_argument(
        "--train_file", type=str, required=True, help="감정분류 학습용 CSV 파일 (columns: text, label)"
    )
    parser.add_argument(
        "--val_file", type=str, required=True, help="감정분류 검증용 CSV 파일 (columns: text, label)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="파인튜닝된 모델 저장 디렉터리"
    )
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--lr", type=float, default=3e-5, help="러닝 레이트")
    parser.add_argument("--max_len", type=int, default=128, help="최대 토큰 길이")
    parser.add_argument("--num_labels", type=int, default=2, help="클래스 수 (감정 분류에서 2개 등)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 토크나이저 로드
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model_dir)

    # 2) 데이터셋/로더 준비
    train_dataset = FinetuningDataset(args.train_file, tokenizer, max_len=args.max_len)
    val_dataset = FinetuningDataset(args.val_file, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3) 모델 불러오기 (프리트레인된 가중치 로드)
    config = TinyBERTConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=args.max_len,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = BERTForSequenceClassification(config, num_labels=args.num_labels).to(device)

    # 프리트레인된 파라미터 로드 (encoder 및 embedding 레이어)
    pretrained_path = os.path.join(args.pretrained_model_dir, "pytorch_model.bin")
    if not os.path.isfile(pretrained_path):
        # epoch 체크포인트 중 최신을 자동으로 찾아 로드해도 됩니다.
        raise FileNotFoundError(f"'{pretrained_path}' 파일이 존재하지 않습니다.")
    state_dict = torch.load(pretrained_path, map_location=device)

    # 키가 mlm 머리 관련 레이어까지 포함되어 있으므로, 필요한 것만 필터링
    new_state = {}
    for k, v in state_dict.items():
        # 'bert.' 로 시작하는 키만 가져옵니다.
        if k.startswith("bert."):
            new_state[k.replace("bert.", "bert.")] = v
    model.bert.load_state_dict(new_state, strict=False)
    print("프리트레인된 인코더 가중치 로드 완료")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4) 파인튜닝 루프
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 검증
        model.eval()
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)

        val_acc = val_correct / val_samples
        print(f"[Epoch {epoch}] Val Acc: {val_acc:.4f}")

        # 가장 좋은 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model.bin")
            torch.save(model.state_dict(), save_path)
            print(f"새로운 최고 성능 모델 저장: {save_path} (Val Acc: {best_val_acc:.4f})")

    # 학습 완료 후 최종 모델 저장
    final_save = os.path.join(args.output_dir, "final_model.bin")
    torch.save(model.state_dict(), final_save)
    print(f"파인튜닝 완료, 최종 모델 저장: {final_save}")


if __name__ == "__main__":
    main()