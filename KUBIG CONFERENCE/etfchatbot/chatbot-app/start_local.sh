#!/bin/bash

# ETF 챗봇 멀티서비스 시작 스크립트 (로컬 환경용)
set -e

echo "=== ETF 챗봇 서비스 시작 (로컬 환경) ==="

# .env 파일 로드
if [ -f ".env" ]; then
    echo ".env 파일 로드 중..."
    export $(cat .env | grep -v '^#' | xargs)
    echo ".env 파일 로드 완료"
else
    echo ".env 파일이 없습니다. env_template.txt를 .env로 복사하고 실제 값을 입력하세요."
fi

# 필요한 디렉토리 미리 생성 (로컬 경로)
echo "필요한 디렉토리 생성 중..."
mkdir -p ./data/.cache/huggingface
mkdir -p ./data/.cache/sentence_transformers
mkdir -p ./data/.aws
echo "디렉토리 생성 완료"

# S3에서 데이터 다운로드 (선택적)
if [ "${SYNC_FROM_S3:-false}" = "true" ]; then
    echo "S3에서 데이터 동기화 중..."
    python -c "
from chatbot.core.s3_utils import download_all_from_s3
try:
    download_all_from_s3()
    print('S3 데이터 동기화 완료')
except Exception as e:
    print(f'S3 동기화 실패 (로컬 모드): {e}')
"
fi

# 환경변수 설정 (로컬 경로)
export PYTHONPATH=$(pwd):$PYTHONPATH

# FastAPI 백그라운드에서 시작
echo "FastAPI 백엔드 시작 중..."
uvicorn app.api.chatbot_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &

FASTAPI_PID=$!

# FastAPI 시작 대기
echo "FastAPI 시작 대기 중..."
for i in {1..30}; do
    if curl -f http://localhost:8000/ping > /dev/null 2>&1; then
        echo "FastAPI 준비 완료"
        break
    fi
    sleep 2
done

# Streamlit 프론트엔드 시작
echo "Streamlit UI 시작 중..."
streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false &

STREAMLIT_PID=$!

# 프로세스 모니터링
echo "서비스 시작 완료 - FastAPI: $FASTAPI_PID, Streamlit: $STREAMLIT_PID"
echo "접속 URL:"
echo "  - FastAPI: http://localhost:8000"
echo "  - Streamlit: http://localhost:8501"

# 종료 신호 처리
cleanup() {
    echo "서비스 종료 중..."
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true
    wait $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true
    echo "모든 서비스가 종료되었습니다."
    exit 0
}

trap cleanup SIGTERM SIGINT

# 무한 대기 (프로세스들이 실행되는 동안)
while kill -0 $FASTAPI_PID 2>/dev/null && kill -0 $STREAMLIT_PID 2>/dev/null; do
    sleep 10
done

echo "서비스 중 하나가 종료되었습니다. 컨테이너를 종료합니다."
cleanup 