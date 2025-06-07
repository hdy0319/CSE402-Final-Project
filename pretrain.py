import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from model import TinyBERTConfig, BERTForPreTraining
from data import PretrainingDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-BERT from-scratch MLM Pretraining with HF Wikipedia")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="학습된 모델과 토크나이저를 저장할 디렉터리"
    )
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=5e-5, help="러닝 레이트")
    parser.add_argument("--max_len", type=int, default=128, help="최대 토큰 길이")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="MLM 마스킹 확률")
    parser.add_argument(
        "--hf_config", type=str, default="20220301.en",
        help="HF Wikipedia dataset config"
    )
    parser.add_argument("--num_samples", type=int, default=10000, help="HF 데이터에서 샘플할 문장 수")
    parser.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="이어 학습할 체크포인트 파일 경로 (예: pretrain_epoch3.pt)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 토크나이저 설정
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(args.output_dir)

    # 2) 데이터셋/데이터로더 준비
    train_dataset = PretrainingDataset(
        tokenizer=tokenizer,
        max_len=args.max_len,
        mlm_probability=args.mlm_probability,
        use_hf=True,
        hf_config=args.hf_config,
        num_samples=args.num_samples,
        seed=args.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    # 3) 모델 및 옵티마이저 설정
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
    model = BERTForPreTraining(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 체크포인트에서 이어 학습
    start_epoch = 1
    if args.resume_from:
        ckpt_path = args.resume_from
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            # 이전 저장 형식에 따라 key 나누기
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_epoch = checkpoint.get('epoch', 1) + 1
            else:
                model.load_state_dict(checkpoint)
            print(f"Checkpoint '{ckpt_path}' 로드, epoch {start_epoch}부터 이어서 학습")
        else:
            print(f"Resume용 체크포인트를 찾을 수 없습니다: {ckpt_path}")

    # 4) 학습 루프
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)

            optimizer.zero_grad()
            _, loss = model(input_ids, attention_mask=attention_mask, mlm_labels=mlm_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % 100 == 0:
                print(f"[Epoch {epoch} | Step {step}/{len(train_loader)}] MLM Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} 완료, 평균 MLM Loss: {avg_loss:.4f}")

        # 체크포인트 저장 (모델+옵티마이저 상태 포함)
        ckpt_path = os.path.join(args.output_dir, f"ckpts/pretrain_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, ckpt_path)
        print(f"모델 체크포인트 저장: {ckpt_path}")

    # 최종 모델 저장
    final_path = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), final_path)
    print(f"최종 모델 저장: {final_path}")


if __name__ == "__main__":
    main()