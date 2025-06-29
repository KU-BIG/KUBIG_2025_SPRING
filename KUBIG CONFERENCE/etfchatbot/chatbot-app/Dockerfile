# Multi-service ETF Chatbot (Streamlit + FastAPI)
FROM python:3.10-slim

# 메타데이터
LABEL maintainer="kubig25ss-agentic-rag"
LABEL version="1.0.0"
LABEL description="ETF Chatbot"

# 환경변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    awscli \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=all -r requirements.txt && \
    pip cache purge && \
    rm -rf /tmp/* && \
    find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10/site-packages -name "__pycache__" -exec rm -rf {} + || true

# 애플리케이션 코드 복사
COPY app/ ./app/
COPY chatbot/ ./chatbot/

# 데이터 디렉토리 생성
RUN mkdir -p data/vectordb

# 포트 노출
EXPOSE 8000 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ping && curl -f http://localhost:8501 || exit 1

# 시작 스크립트 복사
COPY start.sh .
RUN chmod +x start.sh

# non-root 사용자 생성 및 권한 설정
RUN groupadd -r chatbotuser && useradd -r -g chatbotuser chatbotuser && \
    mkdir -p /app/data/vectordb /.aws /.cache && \
    chown -R chatbotuser:chatbotuser /app /.aws /.cache

# 사용자 전환
USER chatbotuser

# 멀티 서비스 시작
CMD ["./start.sh"] 