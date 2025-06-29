import os
import logging
import requests
import mimetypes
import argparse
import re
import time
from datetime import datetime
from typing import List, Dict, Set
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from pymongo import MongoClient
from dotenv import load_dotenv
from src.utils.state_manager import StateManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class MongoDBHandler:
    def __init__(self):
        try:
            # EC2 MongoDB 연결 정보
            EC2_HOST = os.getenv("EC2_HOST")
            EC2_PORT = int(os.getenv("EC2_PORT", "27017"))
            DB_USER = os.getenv("DB_USER")
            DB_PASSWORD = os.getenv("DB_PASSWORD")

            # 디버그를 위한 로깅 추가
            logger.info("=== MongoDB 연결 정보 ===")
            logger.info(f"EC2_HOST: {EC2_HOST}")
            logger.info(f"EC2_PORT: {EC2_PORT}")
            logger.info(f"DB_USER: {DB_USER}")
            logger.info(
                f"DB_PASSWORD: {'*' * len(str(DB_PASSWORD)) if DB_PASSWORD else None}"
            )

            if not all([EC2_HOST, EC2_PORT, DB_USER, DB_PASSWORD]):
                raise ValueError("필수 환경 변수가 설정되지 않았습니다.")

            # MongoDB 연결 URI 구성
            uri = f"mongodb://{DB_USER}:{DB_PASSWORD}@{EC2_HOST}:{EC2_PORT}/?authSource=admin&authMechanism=SCRAM-SHA-1"
            logger.info(f"MongoDB URI: mongodb://{DB_USER}:****@{EC2_HOST}:{EC2_PORT}/")

            # MongoDB 클라이언트 설정
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
                retryReads=True,
                maxPoolSize=1,
            )

            # 연결 테스트
            self.client.admin.command("ping")
            logger.info("MongoDB에 성공적으로 연결되었습니다.")

            self.db = self.client["etf_db"]
            self.collection = self.db["etf_library"]

            # base_dir 설정
            self.base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            # 문서 수 확인
            doc_count = self.collection.count_documents({})
            logger.info(f"etf_library 컬렉션의 전체 문서 수: {doc_count}")

        except Exception as e:
            logger.error(f"MongoDB 연결 실패: {str(e)}")
            raise

    def extract_date_from_url(self, url: str, file_type: str) -> str:
        """
        URL에서 실제 파일 날짜 추출
        
        Args:
            url: 파일 URL
            file_type: 파일 타입 ('price', 'distribution', 'monthly', etc.)
            
        Returns:
            추출된 날짜 문자열 (YYYY-MM-DD 또는 YYYY-MM 형태)
        """
        if not url:
            return ""
            
        try:
            if file_type in ['price', 'distribution']:
                # URL 파라미터에서 gijunYMD 추출
                # 예: gijunYMD=20250529 → 2025-05-29
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                gijun_ymd = query_params.get('gijunYMD', [None])[0]
                
                if gijun_ymd and len(gijun_ymd) == 8:
                    return f"{gijun_ymd[:4]}-{gijun_ymd[4:6]}-{gijun_ymd[6:8]}"
                    
            elif file_type == 'monthly':
                # 월간 보고서: 파일명에서 날짜 추출 후 월 단위로 변환
                # 예: 2ETF01_20250430.pdf → 2025-04
                path = urlparse(url).path
                date_match = re.search(r'(\d{8})\.pdf', path)
                
                if date_match:
                    date_str = date_match.group(1)  # 20250430
                    if len(date_str) == 8:
                        return f"{date_str[:4]}-{date_str[4:6]}"  # 2025-04
                        
        except Exception as e:
            logger.warning(f"URL에서 날짜 추출 실패 ({file_type}): {url} - {e}")
            
        return ""

    def download_etf_docs(self, output_dir: str = "data/etf_raw", limit: int = None, overwrite: bool = False) -> Dict[str, int]:
        """
        ETF 문서 5종(price XLS, distribution XLS, agreement PDF,
        prospectus PDF, monthly PDF)을
        로컬 디렉토리에 ticker별 하위폴더로 다운로드합니다.
        실제 파일의 날짜를 URL에서 추출하여 파일명에 사용합니다.
        중복 URL 및 파일명 체크로 불필요한 다운로드를 방지합니다.
        
        Args:
            output_dir (str): 다운로드 저장 경로
            limit (int, optional): 처리할 ETF 상품(ticker) 수, None이면 모든 상품 처리
            overwrite (bool): 이미 존재하는 파일 덮어쓰기 여부
        """
        out_path = Path(self.base_dir) / output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # StateManager 초기화 (데이터베이스 생성 포함)
        state_manager = StateManager()
        
        # 초기 실행 여부 확인 (etf_database.sqlite가 존재하지 않거나 비어있는 경우)
        is_initial_run = self._check_if_initial_run(state_manager)
        
        if is_initial_run:
            logger.info("=== 초기 실행 감지: 전체 MongoDB 컬렉션 스캔 ===")
            # 전체 문서 조회 (날짜 제한 없음)
            cursor = self.collection.find({}, {"_id": 0, "ticker": 1, "date": 1,
                                               "price_link": 1,
                                               "distribution_link": 1,
                                               "agreement_link":    1,
                                               "prospectus_link":   1,
                                               "monthly_link":      1})
        else:
            logger.info("=== 기존 실행: 최신 날짜 기준 스캔 ===")
            # 기존 로직: 최신 문서만 조회
            cursor = self.collection.find({}, {"_id": 0, "ticker": 1, "date": 1,
                                               "price_link": 1,
                                               "distribution_link": 1,
                                               "agreement_link":    1,
                                               "prospectus_link":   1,
                                               "monthly_link":      1})
        
        # ticker별로 그룹화하고 최신 날짜순으로 정렬
        ticker_docs = {}
        for doc in cursor:
            ticker = doc.get("ticker")
            if not ticker:
                continue
            
            if ticker not in ticker_docs:
                ticker_docs[ticker] = []
            ticker_docs[ticker].append(doc)
        
        # 각 ticker별로 날짜순 정렬 (최신 순)
        for ticker in ticker_docs:
            ticker_docs[ticker].sort(key=lambda x: x.get("date", ""), reverse=True)
        
        # 처리한 고유 ticker 수 추적
        processed_tickers = set()
        
        # 다운로드 통계
        stats = {
            "successful_downloads": 0,
            "failed_downloads": 0,
            "skipped_downloads": 0
        }
        
        for ticker, docs in ticker_docs.items():
            # limit에 도달한 경우 종료
            if limit is not None and len(processed_tickers) >= limit:
                logger.info(f"지정한 limit({limit})에 도달하여 처리 종료")
                break
                
            # 처리 시작 전 ticker 추가
            processed_tickers.add(ticker)
            
            # 초기 실행인 경우 모든 문서 처리, 아니면 최신 문서만 처리
            docs_to_process = docs if is_initial_run else [docs[0]]
            
            for doc_index, doc in enumerate(docs_to_process):
                crawl_date = doc.get("date")  # 크롤링한 날짜
                
                logger.info(f"ETF 처리 중: {ticker} ({len(processed_tickers)}/{limit if limit else '무제한'}) - 크롤링 날짜: {crawl_date}")
                
                # ticker별 디렉토리 생성
                ticker_dir = out_path / ticker
                ticker_dir.mkdir(parents=True, exist_ok=True)
                
                links = {
                    "price":        doc.get("price_link"),
                    "distribution": doc.get("distribution_link"),
                    "agreement":    doc.get("agreement_link"),
                    "prospectus":   doc.get("prospectus_link"),
                    "monthly":      doc.get("monthly_link"),
                }

                # 각 문서 타입별 처리 카운터
                type_counters = {
                    "price": {"downloaded": 0, "skipped": 0, "failed": 0},
                    "distribution": {"downloaded": 0, "skipped": 0, "failed": 0},
                    "agreement": {"downloaded": 0, "skipped": 0, "failed": 0},
                    "prospectus": {"downloaded": 0, "skipped": 0, "failed": 0},
                    "monthly": {"downloaded": 0, "skipped": 0, "failed": 0}
                }

                for dtype, link in links.items():
                    # 링크가 없는 경우 건너뛰기
                    if not link:
                        logger.warning(f"[{ticker} | {crawl_date}] {dtype} 링크 없음, skip")
                        continue

                    # etf_download_states 테이블 기반 URL 중복 체크 (모든 파일 타입 적용)
                    if state_manager.is_etf_file_downloaded(url=link):
                        logger.info(f"[{ticker}] {dtype.upper()} 파일 이미 다운로드됨 (etf_download_states): {link[:80]}...")
                        stats["skipped_downloads"] += 1
                        type_counters[dtype]["skipped"] += 1
                        continue

                    # URL에서 실제 파일 날짜 추출
                    actual_date = self.extract_date_from_url(link, dtype)
                    
                    if actual_date:
                        logger.info(f"[{ticker}] {dtype.upper()} 실제 날짜 추출: {actual_date}")
                        file_date = actual_date
                    else:
                        # 추출 실패시 크롤링 날짜 사용
                        logger.warning(f"[{ticker}] {dtype.upper()} 날짜 추출 실패, 크롤링 날짜 사용: {crawl_date}")
                        file_date = crawl_date

                    # 파일 확장자 결정
                    path = urlparse(link).path
                    ext = Path(path).suffix.lower()

                    # 파일 유형에 따른 확장자 매핑
                    if dtype in ["price", "distribution"]:
                        # price와 distribution은 기본적으로 엑셀 파일이므로 .xls 확장자 사용
                        ext = ".xls"
                    elif dtype in ["agreement", "prospectus", "monthly"]:
                        # 이 파일들은 기본적으로 PDF이므로, 확장자가 없거나 다른 경우 .pdf로 설정
                        if ext not in [".pdf"]:
                            ext = ".pdf"
                    elif not ext:
                        # HEAD 요청으로 Content-Type 확인
                        try:
                            head = requests.head(link, timeout=10)
                            ctype = head.headers.get("Content-Type", "")
                            
                            if "excel" in ctype.lower() or "spreadsheet" in ctype.lower():
                                ext = ".xls"
                            elif "pdf" in ctype.lower():
                                ext = ".pdf"
                            else:
                                ext = mimetypes.guess_extension(ctype.split(";")[0]) or ""
                        except Exception:
                            ext = ""

                    fname = f"{ticker}_{file_date}_{dtype}{ext}"
                    
                    # etf_download_states 기반 파일명 중복 체크 (더 정확한 중복 방지)
                    if state_manager.is_etf_file_downloaded(filename=fname):
                        logger.info(f"[{ticker}] {dtype.upper()} 파일명 이미 다운로드됨 (etf_download_states): {fname}")
                        stats["skipped_downloads"] += 1
                        type_counters[dtype]["skipped"] += 1
                        continue

                    # 실제 다운로드 (ticker별 디렉토리에 저장)
                    try:
                        logger.info(f"[{ticker}] {dtype.upper()} 파일 다운로드 시작: {fname}")
                        resp = requests.get(link, timeout=30)
                        resp.raise_for_status()
                        file_path = ticker_dir / fname
                        file_path.write_bytes(resp.content)
                        logger.info(f"[{ticker}] {dtype.upper()} 다운로드 성공: {fname} ({len(resp.content):,} bytes)")
                        
                        # StateManager에 상태 저장 (매 다운로드마다 즉시 저장 - 모든 파일 타입)
                        state_manager.mark_etf_file_downloaded(
                            url=link,
                            filename=fname,
                            ticker=ticker,
                            doc_type=dtype,
                            date=file_date,
                            file_size=file_path.stat().st_size
                        )
                        logger.debug(f"[{ticker}] {dtype.upper()} 다운로드 상태 저장 완료: etf_download_states 테이블")
                        
                        stats["successful_downloads"] += 1
                        type_counters[dtype]["downloaded"] += 1
                    except requests.exceptions.RequestException as e:
                        logger.error(f"[{ticker}] {dtype.upper()} 다운로드 실패: {fname} — {e}")
                        stats["failed_downloads"] += 1
                        type_counters[dtype]["failed"] += 1
                    except Exception as e:
                        logger.error(f"[{ticker}] {dtype.upper()} 파일 저장 실패: {fname} — {e}")
                        stats["failed_downloads"] += 1
                        type_counters[dtype]["failed"] += 1
                
                # 티커별 처리 결과 요약 로깅
                logger.info(f"[{ticker}] 처리 완료 - 다운로드 결과:")
                for ftype, counts in type_counters.items():
                    total = counts["downloaded"] + counts["skipped"] + counts["failed"]
                    if total > 0:
                        logger.info(f"  {ftype.upper()}: 다운로드={counts['downloaded']}, 건너뜀={counts['skipped']}, 실패={counts['failed']}")
                
        logger.info(f"총 {len(processed_tickers)}개 ETF 상품 처리 완료")

        # 최종 통계 출력 (etf_download_states 기반)
        logger.info("\n=== ETF 문서 다운로드 완료 ===")
        logger.info(f"처리한 ETF 수: {len(processed_tickers)}")
        logger.info(f"성공한 다운로드: {stats['successful_downloads']}")
        logger.info(f"실패한 다운로드: {stats['failed_downloads']}")
        logger.info(f"건너뛴 다운로드: {stats['skipped_downloads']}")
        
        # StateManager 통계 출력 (etf_download_states 테이블 기반)
        download_stats = state_manager.get_etf_download_stats()
        logger.info(f"\n=== StateManager 다운로드 통계 (etf_download_states 테이블) ===")
        logger.info(f"총 다운로드 파일: {download_stats.get('total_files', 0)}")
        logger.info(f"문서 타입별 누적 다운로드:")
        for doc_type, count in download_stats.get('by_doc_type', {}).items():
            logger.info(f"  {doc_type.upper()}: {count}개")
        logger.info(f"티커별 누적 다운로드:")
        for ticker_name, count in download_stats.get('by_ticker', {}).items():
            logger.info(f"  {ticker_name}: {count}개")
        logger.info(f"총 파일 크기: {download_stats.get('total_size_bytes', 0):,} bytes")
        
        return stats

    def _check_if_initial_run(self, state_manager: StateManager) -> bool:
        """
        초기 실행 여부 확인
        
        Returns:
            bool: True if 초기 실행 (etf_database.sqlite가 없거나 etf_download_states 테이블이 비어있음)
        """
        try:
            import sqlite3
            
            # 데이터베이스 파일 존재 여부 확인
            db_path = state_manager.db_path
            if not os.path.exists(db_path):
                logger.info("etf_database.sqlite 파일이 존재하지 않음 - 초기 실행")
                return True
            
            # etf_download_states 테이블의 레코드 수 확인
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # 테이블 존재 여부 확인
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='etf_download_states'
                """)
                
                if not cursor.fetchone():
                    logger.info("etf_download_states 테이블이 존재하지 않음 - 초기 실행")
                    return True
                
                # 테이블이 비어있는지 확인
                cursor.execute("SELECT COUNT(*) FROM etf_download_states")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    logger.info("etf_download_states 테이블이 비어있음 - 초기 실행")
                    return True
                
                logger.info(f"기존 다운로드 레코드 {count}개 발견 - 증분 실행")
                return False
                
        except Exception as e:
            logger.warning(f"초기 실행 여부 확인 중 오류: {e} - 초기 실행으로 간주")
            return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF 문서 다운로드")
    parser.add_argument(
        "--limit",
        type=int,
        help="처리할 ETF 상품(ticker) 수, 지정하지 않으면 모든 상품 처리"
    )
    parser.add_argument(
        "--output-dir",
        default="data/etf_raw",
        help="다운로드 저장 경로 (기본값: data/etf_raw)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 파일 덮어쓰기"
    )
    
    args = parser.parse_args()
    
    with MongoDBHandler() as handler:
        handler.download_etf_docs(
            output_dir=args.output_dir,
            limit=args.limit,
            overwrite=args.overwrite
        )
