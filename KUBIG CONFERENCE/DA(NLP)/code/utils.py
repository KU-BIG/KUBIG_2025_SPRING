# utils.py

import torch
from torch.autograd import Function
import re
import numpy as np

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Feature Extractor로 가는 gradient만 반전
        # Domain Discriminator로 가는 gradient는 그대로 유지 (별도 학습 시)
        # 여기서는 Feature Extractor가 Discriminator를 속이는 방향으로 학습하므로 음수 적용
        output = grad_output.neg() * ctx.alpha
        return output, None

def clean_korean_text(text):
    """
    한국어 텍스트 클리닝 함수.
    - 기본적인 특수문자 제거 (옵션, 문맥에 따라 필요한 특수문자는 유지)
    - 자음/모음 반복 정규화 (예: ㅋㅋㅋ -> ㅋㅋ)
    - 이모티콘 일부 표준화 또는 제거 (옵션)
    - 과도한 공백 제거
    """
    if not isinstance(text, str):
        return ""

    # 1. 한글, 영어, 숫자, 공백, 및 일부 일반적인 구두점 제외하고 제거 (필요에 따라 수정)
    # text = re.sub(r"[^가-힣ㅏ-ㅣㄱ-ㅎa-zA-Z0-9\s.,!?\'\"]", "", text)
    
    # 2. 자음/모음 반복 처리 (예: ㅋㅋㅋ, ㅎㅎㅎ, ㅠㅠㅠ, ㅜㅜㅜ 등)
    # 효과적인 패턴을 위해 더 정교한 접근이 필요할 수 있음
    text = re.sub(r'(ㅋ|ㅎ|ㅜ|ㅠ|ㅡ){3,}', r'\1\1', text) # 3번 이상 반복 -> 2번으로
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text) # 기타 문자 4번 이상 반복 -> 3번으로

    # 3. 간단한 이모티콘 처리 (예: ^^, ;;, 등) - 필요시 추가
    # text = re.sub(r"(\^\^|;;)", " ", text)

    # 4. 과도한 공백 제거 및 표준화
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_grl_alpha(p, schedule_grl_alpha, lambda_adversarial):
    if schedule_grl_alpha:
        # p는 current_iter / total_iters (0에서 1 사이 값)
        return 2. / (1. + np.exp(-5 * p)) - 1
    return lambda_adversarial