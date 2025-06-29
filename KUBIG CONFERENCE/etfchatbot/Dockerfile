FROM continuumio/miniconda3:24.7.1-0 as builder

ARG BUILDKIT_INLINE_CACHE=1
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION=ap-northeast-3
ARG AWS_S3_BUCKET

# AWS credentials를 환경변수로 설정 (런타임에서 사용 가능)
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} 
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
ENV AWS_REGION=${AWS_DEFAULT_REGION}
ENV AWS_S3_BUCKET=${AWS_S3_BUCKET}

WORKDIR /

# pip 설정
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    curl \
    unzip \
    tar \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 기본 의존성 먼저 설치 (순서 중요)
RUN pip install --no-cache-dir -U pip wheel setuptools

# 핵심 패키지 먼저 설치
RUN pip install --no-cache-dir \
    numpy==1.26.2 \
    pandas>=2.2.2 \
    sympy==1.13.3 \
    tenacity==8.3.0

# 나머지 의존성 설치
COPY requirements.txt .
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# 나머지 파일들 복사
COPY . .

# AWS CLI 설치 최적화
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m | sed 's/x86_64/x86_64/;s/aarch64/aarch64/').zip" -o "awscliv2.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update \
    && rm -rf aws awscliv2.zip

# 필요한 파일들만 복사
COPY app ./app
COPY chatbot ./chatbot
COPY data/vectordb ./data/vectordb
COPY src ./src
COPY scripts ./scripts

# 시작 스크립트 생성 (AWS credentials 환경변수 사용)
RUN echo '#!/bin/bash\n\
    echo "AWS 설정 확인: Region=${AWS_REGION}, Bucket=${AWS_S3_BUCKET}"\n\
    aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID}\n\
    aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY}\n\
    aws configure set region ${AWS_REGION}\n\
    aws s3 cp s3://${AWS_S3_BUCKET}/vectordb/chroma.sqlite3 /data/vectordb/chroma.sqlite3 && \
    aws s3 cp s3://${AWS_S3_BUCKET}/vectordb/processed_states.json /data/vectordb/processed_states.json && \
    aws s3 sync s3://${AWS_S3_BUCKET}/vectordb/a9ae4d86-3bee-489e-915a-1824a9eb9e3c/ /data/vectordb/a9ae4d86-3bee-489e-915a-1824a9eb9e3c/ && \
    python -m app.api.chatbot_api' > /app/start.sh && \
    chmod +x /app/start.sh

EXPOSE 8000

CMD ["uvicorn", "app.api.chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"]