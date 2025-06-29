"""
ETF XLS 파일 처리 스크립트

price.xls와 distribution.xls 파일을 처리하여 SQLite 데이터베이스에 저장합니다.
LangChain SQLDatabaseToolkit과 연동하여 LLM이 SQL 쿼리로 데이터에 접근할 수 있게 합니다.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.xls_processor import ETFXLSProcessor
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def process_xls_files(directory: str = "data/etf_raw", 
                     limit: int = None, 
                     db_path: str = "data/etf_database.sqlite") -> bool:
    """
    XLS 파일들을 처리하여 SQLite 데이터베이스에 저장합니다.
    
    Args:
        directory (str): XLS 파일들이 있는 디렉토리
        limit (int, optional): 처리할 ETF 상품 수 제한
        db_path (str): SQLite 데이터베이스 파일 경로
        
    Returns:
        bool: 처리 성공 여부
    """
    try:
        processor = ETFXLSProcessor(db_path)
        success_count = 0
        total_count = 0
        processed_tickers = set()
        
        # 디렉토리가 존재하는지 확인
        if not os.path.exists(directory):
            logger.error(f"디렉토리를 찾을 수 없습니다: {directory}")
            return False
        
        logger.info(f"XLS 파일 처리 시작: {directory}")
        
        # ticker별 디렉토리 스캔
        for ticker_dir in Path(directory).iterdir():
            if not ticker_dir.is_dir():
                continue
                
            ticker = ticker_dir.name
            
            # limit 체크
            if limit is not None and len(processed_tickers) >= limit:
                logger.info(f"지정한 limit({limit})에 도달하여 처리 중단")
                break
                
            processed_tickers.add(ticker)
            logger.info(f"ETF 처리 중: {ticker} ({len(processed_tickers)}/{limit if limit else '무제한'})")
            
            # ticker 디렉토리 내의 XLS 파일들 처리
            xls_files = list(ticker_dir.glob("*.xls")) + list(ticker_dir.glob("*.xlsx"))
            
            for file_path in xls_files:
                try:
                    total_count += 1
                    
                    # 파일명 파싱
                    parsed = processor.parse_filename(file_path.name)
                    if not parsed:
                        logger.warning(f"파일명 파싱 실패: {file_path.name}")
                        continue
                    
                    file_ticker, date, file_type = parsed
                    
                    # ticker 일치 확인
                    if file_ticker != ticker:
                        logger.warning(f"티커 불일치: 폴더({ticker}) vs 파일명({file_ticker})")
                        continue
                    
                    # 이미 처리된 파일인지 확인
                    if processor.is_already_processed(str(file_path)):
                        logger.info(f"이미 처리된 파일: {file_path.name}")
                        continue
                    
                    # 파일 타입별 처리
                    if file_type == "price":
                        result = processor.process_price_file(str(file_path), ticker, date)
                    elif file_type == "distribution":
                        result = processor.process_distribution_file(str(file_path), ticker, date)
                    else:
                        logger.warning(f"알 수 없는 파일 타입: {file_type} ({file_path.name})")
                        continue
                    
                    if result:
                        success_count += 1
                        logger.info(f"파일 처리 성공: {file_path.name}")
                    else:
                        logger.error(f"파일 처리 실패: {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"파일 처리 중 오류 발생 {file_path}: {e}")
                    continue
        
        # 처리 결과 요약
        logger.info("=== XLS 파일 처리 완료 ===")
        logger.info(f"총 처리 파일 수: {total_count}")
        logger.info(f"성공 처리 파일 수: {success_count}")
        logger.info(f"실패 처리 파일 수: {total_count - success_count}")
        logger.info(f"처리된 ETF 상품 수: {len(processed_tickers)}")
        
        # 데이터베이스 상태 요약
        stats = processor.get_processing_stats()
        logger.info("=== 데이터베이스 상태 ===")
        logger.info(f"Price 파일 처리 수: {stats.get('price_files', 0)}")
        logger.info(f"Distribution 파일 처리 수: {stats.get('distribution_files', 0)}")
        logger.info(f"총 가격 레코드 수: {stats.get('total_price_records', 0)}")
        logger.info(f"총 Distribution 레코드 수: {stats.get('total_distribution_records', 0)}")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"XLS 파일 처리 중 전반적 오류: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="ETF XLS 파일 처리")
    parser.add_argument(
        "--directory",
        default="data/etf_raw",
        help="XLS 파일들이 있는 디렉토리 (기본값: data/etf_raw)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="처리할 ETF 상품 수 제한"
    )
    parser.add_argument(
        "--db-path",
        default="data/etf_database.sqlite",
        help="SQLite 데이터베이스 파일 경로"
    )
    
    args = parser.parse_args()
    
    try:
        # XLS 파일 처리
        success = process_xls_files(
            directory=args.directory,
            limit=args.limit,
            db_path=args.db_path
        )
        
        if success:
            logger.info("XLS 파일 처리가 성공적으로 완료되었습니다.")
        else:
            logger.error("XLS 파일 처리 중 오류가 발생했습니다.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 