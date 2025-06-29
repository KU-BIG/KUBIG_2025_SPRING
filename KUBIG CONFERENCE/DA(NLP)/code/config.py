# config.py

import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Configuration
BERT_MODEL_NAME = 'xlm-roberta-base' # 또는 'bert-base-multilingual-cased' 등

# Data Configuration
MAX_LEN = 128
SOURCE_DATA_SAMPLE_SIZE = None # 영어 데이터 샘플링 크기 (None이면 전체)
ADD_SST_TO_SOURCE = True    
USE_SST2_HF = True  
SST_SAMPLE_SIZE = None 
TARGET_DATA_SAMPLE_SIZE = None # 한국어 데이터 샘플링 크기 (None이면 전체)
NAVER_SHOPPING_REVIEW_FILE = 'naver_shopping.txt' # 네이버 쇼핑 리뷰 파일 경로
NAVER_SHOPPING_TRAIN_FILE = 'Train_target.csv'
NAVER_SHOPPING_TEST_FILE = 'Valid_target.csv'


# Training Configuration
BATCH_SIZE = 16 # 메모리 상황에 따라 조절
EPOCHS = 5
LEARNING_RATE = 2e-5
LAMBDA_ADVERSARIAL = 0.1 # DANN의 도메인 손실 가중치 (GRL의 alpha로 직접 사용하거나, 별도 가중치로 사용)
# GRL 알파 스케줄링 사용 여부 (True면 LAMBDA_ADVERSARIAL 대신 p에 따라 alpha 계산)
SCHEDULE_GRL_ALPHA = True

# --- Paths ---
# MODEL_SAVE_PATH = './models/' # 모델 저장 경로 (필요시)