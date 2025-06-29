# scripts/run_pipeline.sh
#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== ETF 데이터 처리 파이프라인 시작 ==="

# 0. ETF 매핑 데이터 업데이트 (MongoDB에서)
echo "0. Updating ETF mapping data from MongoDB..."
python scripts/update_etf_mapping.py

# 1. PDF 처리 (벡터 데이터 포함 - ETF별 컬렉션 자동 생성)
echo "1. Processing PDFs and creating vector embeddings..."
python scripts/process_pdfs.py

# 2. XLS 처리 (ETF 가격 및 분배 데이터)
echo "2. Processing XLS files..."
python scripts/process_xls.py --directory data/etf_raw --db-path data/etf_database.sqlite

# 3. 상태 확인
echo "3. Checking processed states..."
python scripts/check_states.py

echo "=== ETF 데이터 처리 파이프라인 완료 ==="
echo "PDF 문서는 ETF별 컬렉션(etf_069500, etf_226490 등)에 저장되었습니다."

# 4. API 서버 시작 (선택적)
if [ "$1" = "--start-server" ]; then
    echo "4. Starting API server..."
python app/main.py
else
    echo "파이프라인 완료. 서버를 시작하려면 --start-server 옵션을 사용하세요."
fi