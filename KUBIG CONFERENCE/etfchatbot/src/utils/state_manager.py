import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Set, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class StateManager:
    """통합 상태 관리 시스템 - PDF, XLS, ETF raw 파일의 처리 상태를 관리"""
    
    def __init__(self, db_path: str = "data/etf_database.sqlite", 
                 json_path: str = "data/vectordb/processed_states.json"):
        self.db_path = db_path
        self.json_path = json_path
        self._ensure_json_file()
        self._ensure_state_tables()
    
    def _ensure_json_file(self):
        """JSON 파일이 존재하지 않으면 생성"""
        json_file = Path(self.json_path)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        if not json_file.exists():
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    
    def _ensure_state_tables(self):
        """상태 관리용 테이블들이 존재하지 않으면 생성"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # XLS 처리 상태 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS xls_processing_states (
                        file_path TEXT PRIMARY KEY,
                        ticker TEXT NOT NULL,
                        date TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'completed'
                    )
                """)
                
                # ETF raw 파일 다운로드 상태 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS etf_download_states (
                        url TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        doc_type TEXT NOT NULL,
                        date TEXT NOT NULL,
                        downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER
                    )
                """)
                
                conn.commit()
        except Exception as e:
            logger.error(f"상태 테이블 생성 실패: {e}")
    
    # ===== PDF 처리 상태 관리 =====
    
    def load_pdf_states(self) -> Dict:
        """PDF 처리 상태 로드"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"PDF 상태 로드 실패: {e}")
            return {}
    
    def save_pdf_states(self, states: Dict):
        """PDF 처리 상태 저장"""
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(states, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"PDF 상태 저장 실패: {e}")
    
    def is_pdf_processed(self, filename: str, check_type: str = "parsing") -> bool:
        """PDF 파일 처리 여부 확인
        
        Args:
            filename: 확인할 파일명
            check_type: 확인할 처리 타입 ("parsing", "vectorstore", "both")
        """
        states = self.load_pdf_states()
        if filename not in states:
            return False
        
        state = states[filename]
        if check_type == "parsing":
            return state.get("parsing_processed", False)
        elif check_type == "vectorstore":
            return state.get("vectorstore_processed", False)
        elif check_type == "both":
            return (state.get("parsing_processed", False) and 
                   state.get("vectorstore_processed", False))
        return False
    
    def mark_pdf_processed(self, filename: str, state_data: Dict):
        """PDF 파일을 처리됨으로 표시"""
        states = self.load_pdf_states()
        states[filename] = state_data
        self.save_pdf_states(states)
    
    def get_unprocessed_pdfs(self, pdf_files: list, check_type: str = "both") -> list:
        """미처리 PDF 파일 목록 반환"""
        return [pdf for pdf in pdf_files 
                if not self.is_pdf_processed(pdf, check_type)]
    
    # ===== XLS 처리 상태 관리 =====
    
    def is_xls_processed(self, file_path: str) -> bool:
        """XLS 파일 처리 여부 확인"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM xls_processing_states WHERE file_path = ?",
                    (file_path,)
                )
                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"XLS 상태 확인 실패: {e}")
            return False
    
    def mark_xls_processed(self, file_path: str, ticker: str, date: str, file_type: str):
        """XLS 파일을 처리됨으로 표시"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO xls_processing_states 
                    (file_path, ticker, date, file_type) 
                    VALUES (?, ?, ?, ?)
                """, (file_path, ticker, date, file_type))
                conn.commit()
        except Exception as e:
            logger.error(f"XLS 상태 저장 실패: {e}")
    
    def get_xls_processing_stats(self) -> Dict:
        """XLS 처리 통계 반환"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 전체 처리 파일 수
                cursor.execute("SELECT COUNT(*) FROM xls_processing_states")
                total_files = cursor.fetchone()[0]
                
                # 파일 타입별 처리 수
                cursor.execute("""
                    SELECT file_type, COUNT(*) 
                    FROM xls_processing_states 
                    GROUP BY file_type
                """)
                by_type = dict(cursor.fetchall())
                
                # 티커별 처리 수
                cursor.execute("""
                    SELECT ticker, COUNT(*) 
                    FROM xls_processing_states 
                    GROUP BY ticker
                """)
                by_ticker = dict(cursor.fetchall())
                
                # 실제 데이터베이스 레코드 수 확인
                try:
                    cursor.execute("SELECT COUNT(*) FROM etf_prices")
                    total_price_records = cursor.fetchone()[0]
                except:
                    total_price_records = 0
                
                try:
                    cursor.execute("SELECT COUNT(*) FROM etf_distributions")
                    total_distribution_records = cursor.fetchone()[0]
                except:
                    total_distribution_records = 0
                
                # 파일 타입별 실제 처리 파일 수 (price_files, distribution_files)
                price_files = by_type.get('price', 0)
                distribution_files = by_type.get('distribution', 0)
                
                return {
                    "total_files": total_files,
                    "by_type": by_type,
                    "by_ticker": by_ticker,
                    "price_files": price_files,
                    "distribution_files": distribution_files,
                    "total_price_records": total_price_records,
                    "total_distribution_records": total_distribution_records
                }
        except Exception as e:
            logger.error(f"XLS 통계 조회 실패: {e}")
            return {
                "total_files": 0,
                "by_type": {},
                "by_ticker": {},
                "price_files": 0,
                "distribution_files": 0,
                "total_price_records": 0,
                "total_distribution_records": 0
            }
    
    # ===== ETF Raw 파일 다운로드 상태 관리 =====
    
    def is_etf_file_downloaded(self, url: str = None, filename: str = None, 
                             ticker: str = None, doc_type: str = None) -> bool:
        """ETF 파일 다운로드 여부 확인 (URL, 파일명, 또는 ticker+doc_type 조합으로)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if url:
                    cursor.execute(
                        "SELECT COUNT(*) FROM etf_download_states WHERE url = ?",
                        (url,)
                    )
                elif filename:
                    cursor.execute(
                        "SELECT COUNT(*) FROM etf_download_states WHERE filename = ?",
                        (filename,)
                    )
                elif ticker and doc_type:
                    cursor.execute(
                        "SELECT COUNT(*) FROM etf_download_states WHERE ticker = ? AND doc_type = ?",
                        (ticker, doc_type)
                    )
                else:
                    return False
                
                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"ETF 다운로드 상태 확인 실패: {e}")
            return False

    def mark_etf_file_downloaded(self, url: str, filename: str, ticker: str, 
                                doc_type: str, date: str, file_size: int = None):
        """ETF 파일을 다운로드됨으로 표시 (모든 파일 타입 지원)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO etf_download_states 
                    (url, filename, ticker, doc_type, date, file_size) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (url, filename, ticker, doc_type, date, file_size))
                conn.commit()
                logger.debug(f"ETF 파일 다운로드 상태 저장: {doc_type.upper()} - {ticker} - {filename}")
        except Exception as e:
            logger.error(f"ETF 다운로드 상태 저장 실패: {e}")

    def get_downloaded_urls(self) -> Set[str]:
        """다운로드된 URL 집합 반환 (모든 파일 타입)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT url FROM etf_download_states")
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"다운로드된 URL 조회 실패: {e}")
            return set()
    
    def get_downloaded_filenames(self) -> Set[str]:
        """다운로드된 파일명 집합 반환 (모든 파일 타입)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT filename FROM etf_download_states")
                return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"다운로드된 파일명 조회 실패: {e}")
            return set()

    def get_downloaded_files_by_type(self, doc_type: str = None) -> Dict:
        """특정 문서 타입의 다운로드된 파일 목록 반환"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if doc_type:
                    cursor.execute("""
                        SELECT ticker, filename, date, downloaded_at, file_size
                        FROM etf_download_states 
                        WHERE doc_type = ?
                        ORDER BY downloaded_at DESC
                    """, (doc_type,))
                else:
                    cursor.execute("""
                        SELECT doc_type, ticker, filename, date, downloaded_at, file_size
                        FROM etf_download_states 
                        ORDER BY doc_type, downloaded_at DESC
                    """)
                
                results = cursor.fetchall()
                
                if doc_type:
                    return {
                        "doc_type": doc_type,
                        "files": [
                            {
                                "ticker": row[0],
                                "filename": row[1], 
                                "date": row[2],
                                "downloaded_at": row[3],
                                "file_size": row[4]
                            } for row in results
                        ],
                        "total_count": len(results)
                    }
                else:
                    by_type = {}
                    for row in results:
                        dtype = row[0]
                        if dtype not in by_type:
                            by_type[dtype] = []
                        by_type[dtype].append({
                            "ticker": row[1],
                            "filename": row[2],
                            "date": row[3], 
                            "downloaded_at": row[4],
                            "file_size": row[5]
                        })
                    return by_type
                    
        except Exception as e:
            logger.error(f"문서 타입별 다운로드 파일 조회 실패: {e}")
            return {}

    def check_etf_file_integrity(self, ticker: str = None) -> Dict:
        """ETF 파일의 무결성 확인 (모든 문서 타입이 다운로드되었는지)"""
        expected_types = {"price", "distribution", "agreement", "prospectus", "monthly"}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if ticker:
                    # 특정 티커의 파일 타입별 다운로드 상태
                    cursor.execute("""
                        SELECT doc_type, COUNT(*) as count, MAX(date) as latest_date
                        FROM etf_download_states 
                        WHERE ticker = ?
                        GROUP BY doc_type
                    """, (ticker,))
                    
                    results = dict(cursor.fetchall())
                    
                    integrity_report = {
                        "ticker": ticker,
                        "downloaded_types": set(results.keys()),
                        "missing_types": expected_types - set(results.keys()),
                        "complete": len(set(results.keys())) == len(expected_types),
                        "details": {}
                    }
                    
                    for doc_type in expected_types:
                        if doc_type in results:
                            integrity_report["details"][doc_type] = {
                                "status": "downloaded",
                                "count": results[doc_type][0], 
                                "latest_date": results[doc_type][1]
                            }
                        else:
                            integrity_report["details"][doc_type] = {
                                "status": "missing",
                                "count": 0,
                                "latest_date": None
                            }
                    
                    return integrity_report
                    
                else:
                    # 전체 티커별 무결성 확인
                    cursor.execute("""
                        SELECT ticker, doc_type, COUNT(*) as count
                        FROM etf_download_states 
                        GROUP BY ticker, doc_type
                    """)
                    
                    results = cursor.fetchall()
                    ticker_data = {}
                    
                    for ticker_name, doc_type, count in results:
                        if ticker_name not in ticker_data:
                            ticker_data[ticker_name] = {}
                        ticker_data[ticker_name][doc_type] = count
                    
                    integrity_summary = {}
                    for ticker_name, types_data in ticker_data.items():
                        downloaded_types = set(types_data.keys())
                        missing_types = expected_types - downloaded_types
                        
                        integrity_summary[ticker_name] = {
                            "downloaded_types": downloaded_types,
                            "missing_types": missing_types,
                            "complete": len(missing_types) == 0,
                            "completion_rate": len(downloaded_types) / len(expected_types) * 100
                        }
                    
                    return integrity_summary
                    
        except Exception as e:
            logger.error(f"ETF 파일 무결성 확인 실패: {e}")
            return {}
    
    def get_etf_download_stats(self) -> Dict:
        """ETF 다운로드 통계 반환 (모든 파일 타입별 상세 정보 포함)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 전체 다운로드 파일 수
                cursor.execute("SELECT COUNT(*) FROM etf_download_states")
                total_files = cursor.fetchone()[0]
                
                # 문서 타입별 다운로드 수 (상세)
                cursor.execute("""
                    SELECT doc_type, COUNT(*) as count, 
                           MIN(downloaded_at) as first_download,
                           MAX(downloaded_at) as last_download,
                           SUM(file_size) as total_size
                    FROM etf_download_states 
                    GROUP BY doc_type
                    ORDER BY doc_type
                """)
                by_doc_type_detailed = {}
                by_doc_type = {}
                
                for row in cursor.fetchall():
                    doc_type, count, first_dl, last_dl, total_size = row
                    by_doc_type[doc_type] = count
                    by_doc_type_detailed[doc_type] = {
                        "count": count,
                        "first_download": first_dl,
                        "last_download": last_dl,
                        "total_size_bytes": total_size or 0
                    }
                
                # 티커별 다운로드 수
                cursor.execute("""
                    SELECT ticker, COUNT(*) as total_files,
                           COUNT(DISTINCT doc_type) as unique_types,
                           MAX(downloaded_at) as last_download
                    FROM etf_download_states 
                    GROUP BY ticker
                    ORDER BY ticker
                """)
                by_ticker_detailed = {}
                by_ticker = {}
                
                for row in cursor.fetchall():
                    ticker, total_files, unique_types, last_dl = row
                    by_ticker[ticker] = total_files
                    by_ticker_detailed[ticker] = {
                        "total_files": total_files,
                        "unique_doc_types": unique_types,
                        "last_download": last_dl,
                        "completion_rate": (unique_types / 5) * 100  # 5개 문서 타입 기준
                    }
                
                # 총 파일 크기
                cursor.execute("SELECT SUM(file_size) FROM etf_download_states WHERE file_size IS NOT NULL")
                total_size = cursor.fetchone()[0] or 0
                
                # 최근 다운로드 활동
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM etf_download_states 
                    WHERE datetime(downloaded_at) >= datetime('now', '-1 day')
                """)
                recent_downloads = cursor.fetchone()[0]
                
                return {
                    "total_files": total_files,
                    "by_doc_type": by_doc_type,
                    "by_doc_type_detailed": by_doc_type_detailed,
                    "by_ticker": by_ticker,
                    "by_ticker_detailed": by_ticker_detailed,
                    "total_size_bytes": total_size,
                    "recent_downloads_24h": recent_downloads,
                    "expected_doc_types": ["price", "distribution", "agreement", "prospectus", "monthly"]
                }
        except Exception as e:
            logger.error(f"ETF 다운로드 통계 조회 실패: {e}")
            return {}
    
    # ===== 통합 상태 조회 =====
    
    def get_overall_stats(self) -> Dict:
        """전체 처리 상태 통계 반환"""
        pdf_states = self.load_pdf_states()
        xls_stats = self.get_xls_processing_stats()
        etf_stats = self.get_etf_download_stats()
        
        # PDF 상태 분석
        pdf_parsing_processed = sum(1 for state in pdf_states.values() 
                                  if state.get("parsing_processed", False))
        pdf_vectorstore_processed = sum(1 for state in pdf_states.values() 
                                      if state.get("vectorstore_processed", False))
        
        return {
            "pdf": {
                "total_files": len(pdf_states),
                "parsing_processed": pdf_parsing_processed,
                "vectorstore_processed": pdf_vectorstore_processed
            },
            "xls": xls_stats,
            "etf_downloads": etf_stats,
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_states(self, state_type: str = "all"):
        """상태 초기화 (개발/테스트용)
        
        Args:
            state_type: "pdf", "xls", "etf", "all"
        """
        if state_type in ["pdf", "all"]:
            self.save_pdf_states({})
            logger.info("PDF 상태 초기화됨")
        
        if state_type in ["xls", "etf", "all"]:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    if state_type in ["xls", "all"]:
                        cursor.execute("DELETE FROM xls_processing_states")
                        logger.info("XLS 상태 초기화됨")
                    
                    if state_type in ["etf", "all"]:
                        cursor.execute("DELETE FROM etf_download_states")
                        logger.info("ETF 다운로드 상태 초기화됨")
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"상태 초기화 실패: {e}") 