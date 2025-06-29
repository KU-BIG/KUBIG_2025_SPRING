# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config # config.py 임포트
from dataset import SentimentDataset, load_naver_shopping_reviews, load_imdb_data # dataset.py 임포트
from model import FeatureExtractor, SentimentClassifier, DomainDiscriminator # model.py 임포트
from trainer import train_epoch, evaluate_target # trainer.py 임포트

def main():
    print(f"Using device: {config.DEVICE}")

    # --- 1. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    # --- 2. Load Data ---
    print("Loading source (English) data...")
    # IMDB 전체 데이터 로드
    source_texts, source_labels = load_imdb_data(sample_size=config.SOURCE_DATA_SAMPLE_SIZE) # sample_size=None 전달
    print(f"Loaded {len(source_texts)} IMDB samples (full dataset).")

    if config.ADD_SST_TO_SOURCE:
        print("Attempting to load SST data...")
        sst_texts, sst_labels = [], []
        if config.USE_SST2_HF:
            # SST 전체 데이터 로드 (주로 train 스플릿 사용)
            sst_texts, sst_labels = load_sst2_data(sample_size=config.SST_SAMPLE_SIZE, split='train') # sample_size=None 전달
        else:
            # (파일 직접 파싱 로직 - 필요시 sample_size=None 처리)
            pass 
        
        if sst_texts:
            source_texts.extend(sst_texts)
            source_labels.extend(sst_labels) # 이전 버그 수정됨
            print(f"Added {len(sst_texts)} SST samples (full dataset). Total source samples: {len(source_texts)}")
    
    source_dataset = SentimentDataset(source_texts, source_labels, tokenizer, config.MAX_LEN, is_source=True)
    source_loader = DataLoader(source_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True) # drop_last=True 유지 또는 False 고려

    # --- 타겟(한국어) 데이터 로드 (학습용/평가용 파일 별도 사용) ---
    print("Loading target (Korean) training data...")
    target_train_file_path = f"{project_path}/{config.NAVER_SHOPPING_TRAIN_FILE}"
    target_train_texts, target_train_labels = load_naver_shopping_reviews(
        target_train_file_path,
        sample_size=config.TARGET_DATA_SAMPLE_SIZE # sample_size=None 전달 (파일 전체 사용)
    )
    print(f"Loaded {len(target_train_texts)} target training samples.")
    target_train_dataset = SentimentDataset(target_train_texts, target_train_labels, tokenizer, config.MAX_LEN, is_source=False)
    target_train_loader = DataLoader(target_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True) # drop_last=True 유지 또는 False 고려

    print("Loading target (Korean) test data...")
    target_test_file_path = f"{project_path}/{config.NAVER_SHOPPING_TEST_FILE}"
    target_test_texts, target_test_labels = load_naver_shopping_reviews(
        target_test_file_path,
        sample_size=config.TARGET_DATA_SAMPLE_SIZE # sample_size=None 전달 (파일 전체 사용)
    )
    print(f"Loaded {len(target_test_texts)} target test samples.")
    target_test_dataset = SentimentDataset(target_test_texts, target_test_labels, tokenizer, config.MAX_LEN, is_source=False)
    target_test_loader = DataLoader(target_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False) # 테스트 로더는 shuffle=False

    # --- (모델 초기화, 옵티마이저, 학습 루프 등 나머지 코드는 거의 동일하게 유지) ---
    # len_total_iters_epoch 계산 시 주의:
    # source_loader와 target_train_loader의 길이가 다를 수 있음 (학습 시 짧은 쪽 기준)
    if source_loader and target_train_loader: # 두 로더가 모두 준비되었는지 확인
        len_total_iters_epoch = min(len(source_loader), len(target_train_loader))
    else: # 하나라도 비어있으면 학습 불가 또는 다른 처리 필요
        print("Error: Source or Target train loader is not available. Cannot determine iterations per epoch.")
        return None, None, None, None, None, None, None, None # 함수 시그니처에 맞게 반환값 조정


    # --- 3. Initialize Models ---
    feature_extractor = FeatureExtractor(config.BERT_MODEL_NAME).to(config.DEVICE)
    sentiment_classifier = SentimentClassifier(feature_extractor.bert_output_dim).to(config.DEVICE)
    domain_discriminator = DomainDiscriminator(feature_extractor.bert_output_dim).to(config.DEVICE)

    # --- 4. Optimizer and Loss Functions ---
    # 모든 모델 파라미터를 하나의 옵티마이저로 학습
    optimizer = optim.AdamW(
        list(feature_extractor.parameters()) +
        list(sentiment_classifier.parameters()) +
        list(domain_discriminator.parameters()),
        lr=config.LEARNING_RATE
    )
    sentiment_criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    domain_criterion = nn.BCEWithLogitsLoss().to(config.DEVICE) # Discriminator 출력이 logit이라고 가정

    # --- 5. Training Loop ---
    print("Starting DANN training...")
    for epoch in range(config.EPOCHS):
        print(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        avg_sent_loss, avg_dom_loss, avg_comb_loss = train_epoch(
            feature_extractor, sentiment_classifier, domain_discriminator,
            source_loader, target_train_loader, optimizer,
            sentiment_criterion, domain_criterion, config.DEVICE,
            epoch, config.EPOCHS, config.LAMBDA_ADVERSARIAL, 
            config.SCHEDULE_GRL_ALPHA, len_total_iters_epoch
        )
        print(f"Epoch {epoch+1} Avg Losses: Sent={avg_sent_loss:.4f}, Dom={avg_dom_loss:.4f}, Comb={avg_comb_loss:.4f}")
        
        # Evaluate on target test set
        evaluate_target(feature_extractor, sentiment_classifier, target_test_loader, config.DEVICE, epoch)

    print("Training finished.")
    # (Optional: Save models)
    # torch.save(feature_extractor.state_dict(), f"{config.MODEL_SAVE_PATH}feature_extractor_final.pth")
    # torch.save(sentiment_classifier.state_dict(), f"{config.MODEL_SAVE_PATH}sentiment_classifier_final.pth")

if __name__ == '__main__':
    # (Optional: Create model save directory if it doesn't exist)
    # import os
    # if config.MODEL_SAVE_PATH and not os.path.exists(config.MODEL_SAVE_PATH):
    #     os.makedirs(config.MODEL_SAVE_PATH)
    main()