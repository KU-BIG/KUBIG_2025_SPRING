# model.py

import torch
import torch.nn as nn
from transformers import AutoModel
import utils
from utils import GradientReversalFn # utils.py의 GRL 임포트

class FeatureExtractor(nn.Module):
    def __init__(self, model_name, finetune_bert=True):
        super(FeatureExtractor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if not finetune_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.bert_output_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] # [CLS] token
        return pooled_output

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout_rate=0.3):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(input_dim // 2, num_classes)
        
    def forward(self, features):
        x = self.dropout(self.relu(self.fc1(features)))
        x = self.fc2(x)
        return x # Raw logits

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(input_dim // 2, 1) # Output a single logit for domain (source vs target)

    def forward(self, features_with_grl): # 특징은 GRL을 통과한 후 입력됨
        x = self.dropout(self.relu(self.fc1(features_with_grl)))
        x = self.fc2(x)
        return x # Raw logits for BCEWithLogitsLoss