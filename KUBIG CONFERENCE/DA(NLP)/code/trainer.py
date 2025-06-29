# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from utils import GradientReversalFn, get_grl_alpha # utils.py의 함수 임포트
import numpy as np # get_grl_alpha에서 사용

def train_epoch(feature_extractor, sentiment_classifier, domain_discriminator,
                source_loader, target_loader, optimizer, 
                sentiment_criterion, domain_criterion, 
                device, current_epoch, total_epochs, 
                lambda_adversarial, schedule_grl_alpha, len_total_iters_epoch):
    
    feature_extractor.train()
    sentiment_classifier.train()
    domain_discriminator.train()

    total_sentiment_loss = 0
    total_domain_loss = 0
    total_combined_loss = 0
    
    # 데이터로더 길이 중 짧은 것을 기준으로 이터레이션
    # 또는 itertools.cycle 사용 가능
    len_dataloader = min(len(source_loader), len(target_loader))
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for batch_idx in range(len_dataloader):
        # --- GRL Alpha Scheduling ---
        # p는 전체 학습 과정 중 현재 진행도 (0에서 1)
        p_progress = float(batch_idx + current_epoch * len_dataloader) / (total_epochs * len_dataloader)
        grl_alpha = get_grl_alpha(p_progress, schedule_grl_alpha, lambda_adversarial)

        # --- Source Data ---
        s_batch = next(source_iter)
        s_input_ids = s_batch['input_ids'].to(device)
        s_attention_mask = s_batch['attention_mask'].to(device)
        s_sentiment_labels = s_batch['sentiment_labels'].to(device)
        s_domain_labels = s_batch['domain_labels'].float().unsqueeze(1).to(device) # For BCEWithLogitsLoss

        # --- Target Data ---
        # 타겟 데이터의 감성 레이블은 Sentiment Classifier 학습에 사용되지 않음.
        t_batch = next(target_iter)
        t_input_ids = t_batch['input_ids'].to(device)
        t_attention_mask = t_batch['attention_mask'].to(device)
        # t_sentiment_labels = t_batch['sentiment_labels'].to(device) # 사용 안함
        t_domain_labels = t_batch['domain_labels'].float().unsqueeze(1).to(device) # For BCEWithLogitsLoss
        
        optimizer.zero_grad()

        # --- Feature Extraction ---
        s_features = feature_extractor(s_input_ids, s_attention_mask)
        t_features = feature_extractor(t_input_ids, t_attention_mask)

        # --- 1. Sentiment Classification Loss (Source Domain Only) ---
        s_sentiment_preds = sentiment_classifier(s_features)
        loss_sentiment = sentiment_criterion(s_sentiment_preds, s_sentiment_labels)

        # --- 2. Domain Discrimination Loss (Source and Target) ---
        # GRL을 통과한 특징을 Domain Discriminator에 입력
        s_features_grl = GradientReversalFn.apply(s_features, grl_alpha)
        t_features_grl = GradientReversalFn.apply(t_features, grl_alpha)
        
        s_domain_preds = domain_discriminator(s_features_grl)
        t_domain_preds = domain_discriminator(t_features_grl)

        # Domain Discriminator는 실제 도메인 레이블을 정확히 맞추도록 학습
        # Source 도메인 레이블은 0, Target 도메인 레이블은 1
        loss_domain_s = domain_criterion(s_domain_preds, s_domain_labels) 
        loss_domain_t = domain_criterion(t_domain_preds, t_domain_labels) 
        loss_domain = loss_domain_s + loss_domain_t
        
        # --- Combined Loss for Feature Extractor & Sentiment Classifier ---
        # Feature Extractor는 Sentiment Loss는 최소화, Domain Loss는 (GRL을 통해) 최대화 (즉, Discriminator를 혼동시키도록)
        # LAMBDA_ADVERSARIAL은 Domain Loss의 가중치 역할
        combined_loss = loss_sentiment + lambda_adversarial * loss_domain
        
        combined_loss.backward()
        optimizer.step()

        total_sentiment_loss += loss_sentiment.item()
        total_domain_loss += loss_domain.item()
        total_combined_loss += combined_loss.item()

        if batch_idx > 0 and batch_idx % (len_dataloader // 5 + 1) == 0: # 로그 빈도 조절
            print(f"  Epoch {current_epoch+1}, Batch {batch_idx}/{len_dataloader}, GRL alpha: {grl_alpha:.4f}")
            print(f"  SentLoss: {loss_sentiment.item():.4f}, DomLoss: {loss_domain.item():.4f} (weight: {lambda_adversarial}), CombLoss: {combined_loss.item():.4f}")
    
    avg_sent_loss = total_sentiment_loss / len_dataloader
    avg_dom_loss = total_domain_loss / len_dataloader
    avg_comb_loss = total_combined_loss / len_dataloader
    return avg_sent_loss, avg_dom_loss, avg_comb_loss


def evaluate_target(feature_extractor, sentiment_classifier, target_test_loader, device, epoch_num):
    feature_extractor.eval()
    sentiment_classifier.eval()

    all_sentiment_labels = []
    all_sentiment_preds = []
    
    with torch.no_grad():
        for batch in target_test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device) # 타겟의 실제 감성 레이블

            features = feature_extractor(input_ids, attention_mask)
            sentiment_preds_logits = sentiment_classifier(features)
            
            _, predicted_sentiment = torch.max(sentiment_preds_logits, dim=1)
            
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())
            all_sentiment_preds.extend(predicted_sentiment.cpu().numpy())
    
    if not all_sentiment_labels or not all_sentiment_preds:
        print(f"Epoch {epoch_num+1} - Target (Korean) Test: No data to evaluate.")
        return 0.0

    print(f"\n--- Epoch {epoch_num+1} - Target (Korean) Test Results ---")
    report = classification_report(all_sentiment_labels, all_sentiment_preds, 
                                   target_names=['부정 (0)', '긍정 (1)'], zero_division=0, output_dict=True)
    accuracy = report['accuracy']
    f1_macro = report['macro avg']['f1-score']
    print(classification_report(all_sentiment_labels, all_sentiment_preds, 
                                   target_names=['부정 (0)', '긍정 (1)'], zero_division=0))
    print(f"Target Accuracy: {accuracy:.4f}, Macro F1-score: {f1_macro:.4f}")
    print("---------------------------------------------------\n")
    return accuracy