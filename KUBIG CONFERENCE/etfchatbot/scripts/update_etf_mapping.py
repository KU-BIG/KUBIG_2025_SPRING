#!/usr/bin/env python3
"""
ETF 매핑 데이터 업데이트 스크립트

MongoDB의 etf_db/etf_library 컬렉션에서 ETF 매핑 데이터를 가져와서
data/etf_mapping.json 파일로 저장합니다.

Jenkins 환경에서는 반드시 MongoDB에 크롤링된 실제 데이터가 있어야 합니다.
"""

import sys
import os
import logging

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etf_mapping import ETFMappingManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """ETF 매핑 데이터 업데이트 메인 함수"""
    
    # MongoDB 연결 정보
    mongo_host = os.environ.get('MONGODB_HOST', 'localhost')
    mongo_port = int(os.environ.get('MONGODB_PORT', '27017'))
    
    logger.info(f"MongoDB 연결 시도: {mongo_host}:{mongo_port}")
    
    # ETF 매핑 매니저 초기화
    manager = ETFMappingManager(mongo_host, mongo_port)
    
    try:
        # MongoDB에서 ETF 매핑 데이터 업데이트 (필수)
        success = manager.update_mapping_from_mongodb()
        
        if success:
            logger.info("ETF 매핑 데이터 업데이트 성공")
            
            # 업데이트된 매핑 정보 확인
            mapping = manager.load_mapping_from_file()
            logger.info(f"총 {len(mapping)}개의 ETF 매핑 데이터가 저장되었습니다")
            
            # 일부 샘플 출력
            sample_count = min(10, len(mapping))
            sample_items = list(mapping.items())[:sample_count]
            
            logger.info("업데이트된 ETF 매핑 샘플:")
            for ticker, etf_name in sample_items:
                logger.info(f"  {ticker}: {etf_name}")
                
            if len(mapping) > sample_count:
                logger.info(f"  ... (외 {len(mapping) - sample_count}개)")
            
            return True
            
        else:
            logger.error("MongoDB에서 ETF 매핑 데이터 업데이트 실패")
            logger.error("Jenkins 환경에서는 반드시 MongoDB에 크롤링된 실제 ETF 데이터가 필요합니다")
            logger.error("먼저 ETF 크롤링을 실행하여 MongoDB에 데이터를 저장하세요")
            return False
            
    except Exception as e:
        logger.error(f"ETF 매핑 업데이트 중 오류 발생: {e}")
        logger.error("MongoDB 연결 또는 데이터 접근에 실패했습니다")
        logger.error("Jenkins 환경에서는 ETF 크롤링이 먼저 성공적으로 완료되어야 합니다")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 