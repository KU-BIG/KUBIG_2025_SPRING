# dataset.py

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils import clean_korean_text # utils.py의 함수 임포트
from datasets import load_dataset as hf_load_dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, is_source=True):
        self.texts = texts
        # is_source가 False(타겟 도메인)일 경우, sentiment label은 평가에만 사용됨
        self.sentiment_labels = labels # 실제 감성 레이블
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_source = is_source # 도메인 레이블 생성에 사용

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # 타겟 도메인이라도 sentiment_labels는 전달 (평가 시 사용 위함)
        sentiment_label = self.sentiment_labels[idx] if self.sentiment_labels is not None else -1 # DANN 학습 시 타겟의 sentiment_label은 loss 계산에 직접 사용 안함

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # DANN 학습 시 도메인 판별기가 사용할 도메인 레이블
        # Source domain: 0, Target domain: 1
        domain_label = 0 if self.is_source else 1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment_labels': torch.tensor(sentiment_label, dtype=torch.long),
            'domain_labels': torch.tensor(domain_label, dtype=torch.long)
        }

def load_naver_shopping_reviews(file_path, sample_size=None, random_state=42):
    try:
        # ... (파일 읽기 및 초기 처리 로직은 이전과 동일) ...
        print(f"Attempting to load: {file_path}")
        df = pd.read_csv(file_path, 
                         sep=',',
                         header=0, 
                         on_bad_lines='skip')
        print(f"Initial rows: {len(df)}, Columns: {df.columns.tolist()}")
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Columns 'label' and 'text' not found. Actual columns: {df.columns.tolist()}")

        df['text'] = df['text'].fillna('').astype(str)
        df['text'] = df['text'].apply(clean_korean_text)
        df = df[df['text'].str.strip().str.len() > 0]
        
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        # df = df[df['label'].isin([0, 1])] # 필요시 유지

        if df.empty:
            print(f"Warning: No data left after processing for file: {file_path}")
            return [], []

        if sample_size and sample_size < len(df):
            # --- 수정된 부분: stratify 인자 제거 ---
            df = df.sample(n=sample_size, 
                           random_state=random_state) # stratify 인자 제거
            # ------------------------------------
            
        print(f"Loaded and processed {len(df)} samples from {file_path}.")
        return df['text'].tolist(), df['label'].tolist()

    # ... (except 절은 이전과 동일) ...
    except FileNotFoundError:
        print(f"'{file_path}' 파일을 찾을 수 없습니다. 임시 한국어 데이터를 사용합니다.")
        num_samples = sample_size if sample_size else 2
        temp_texts = [clean_korean_text('너무 좋아요 강추!') if i % 2 == 0 else clean_korean_text('이건 좀 별로네요 비추천.') for i in range(num_samples)]
        temp_labels = [1 if i % 2 == 0 else 0 for i in range(num_samples)]
        return temp_texts, temp_labels
    except Exception as e:
        print(f"Error loading or processing reviews from '{file_path}': {e}")
        import traceback
        traceback.print_exc()
        num_samples = sample_size if sample_size else 2
        temp_texts = [f"Error processing data {i}" for i in range(num_samples)]
        temp_labels = [0 for i in range(num_samples)]
        return temp_texts, temp_labels

def load_imdb_data(sample_size=None, random_state=42):
    try:
        from datasets import load_dataset
        imdb_dataset = load_dataset("imdb")
        
        train_df = imdb_dataset['train'].to_pandas()
        # test_df = imdb_dataset['test'].to_pandas() # DANN에서는 주로 source의 train만 사용
        
        # 영어 텍스트도 간단한 클리닝 (옵션)
        # train_df['text'] = train_df['text'].apply(some_english_cleaning_function)

        if sample_size and sample_size < len(train_df):
            train_df = train_df.sample(n=sample_size, random_state=random_state)
            
        return train_df['text'].tolist(), train_df['label'].tolist()
    except (ImportError, FileNotFoundError) as e:
        print(f"HuggingFace datasets 또는 IMDB 로드 실패: {e}. 임시 영어 데이터를 사용합니다.")
        num_samples = sample_size if sample_size else 100
        temp_texts = [f"This is a great product review number {i}" if i % 2 == 0 else f"A very bad experience for item {i}" for i in range(num_samples)]
        temp_labels = [1 if i % 2 == 0 else 0 for i in range(num_samples)]
        return temp_texts, temp_labels


def load_sst2_data(sample_size=None, random_state=42, split='train'):
    """Hugging Face datasets 라이브러리를 사용하여 SST-2 데이터를 로드합니다."""
    try:
        dataset = hf_load_dataset("glue", "sst2")
        
        if split not in dataset:
            print(f"SST-2에 '{split}' 스플릿이 없습니다. 사용 가능한 스플릿: {list(dataset.keys())}")
            return [], []
            
        df = dataset[split].to_pandas()
        
        # SST-2는 이미 label이 0 (부정), 1 (긍정)으로 되어 있음
        # 'sentence' 컬럼과 'label' 컬럼 사용
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state)
            
        texts = df['sentence'].tolist()
        labels = df['label'].tolist()
        
        print(f"Loaded {len(texts)} samples from SST-2 '{split}' split.")
        return texts, labels
    except Exception as e:
        print(f"HuggingFace SST-2 데이터 로드 실패: {e}. 임시 데이터를 반환합니다.")
        num_samples = sample_size if sample_size else 100
        temp_texts = [f"This SST-2 sentence is great {i}" if i % 2 == 0 else f"This SST-2 sentence is bad {i}" for i in range(num_samples)]
        temp_labels = [1 if i % 2 == 0 else 0 for i in range(num_samples)]
        return temp_texts, temp_labels