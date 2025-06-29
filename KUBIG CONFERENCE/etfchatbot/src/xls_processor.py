import os
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re
from src.utils.state_manager import StateManager
from src.utils.etf_utils import get_etf_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETFXLSProcessor:
    def __init__(self, db_path: str = "data/etf_database.sqlite"):
        """
        ETF XLS 파일 프로세서 초기화 - 완전한 정보 저장
        
        Args:
            db_path (str): SQLite 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 데이터베이스 디렉토리 생성
        db_dir = os.path.dirname(os.path.join(self.base_dir, db_path))
        os.makedirs(db_dir, exist_ok=True)
        
        self.db_full_path = os.path.join(self.base_dir, db_path)
        self.state_manager = StateManager(db_path=self.db_full_path)
        self._create_database()
    
    def _create_database(self):
        """데이터베이스와 테이블 생성 - 완전한 정보 저장"""
        try:
            with sqlite3.connect(self.db_full_path) as conn:
                cursor = conn.cursor()
                
                # 완전한 정보를 담는 새로운 테이블들
                self._create_comprehensive_tables(cursor)
                
                conn.commit()
                logger.info("완전한 정보 저장용 데이터베이스 테이블 생성 완료")
                
        except Exception as e:
            logger.error(f"데이터베이스 생성 실패: {e}")
            raise
    
    def _create_comprehensive_tables(self, cursor):
        """모든 XLS 정보를 포함하는 완전한 테이블 생성"""
        
        # ETF 가격 정보 테이블 - 모든 price.xls 정보 포함
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_prices (
                ticker TEXT NOT NULL,
                etf_name TEXT,
                date TEXT NOT NULL,
                closing_price REAL,
                price_change REAL,
                price_change_rate REAL,
                volume INTEGER,
                nav REAL,
                nav_change REAL,
                nav_change_rate REAL,
                tax_base_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # ETF 구성종목 정보 테이블 - 모든 distribution.xls 정보 포함
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_distributions (
                ticker TEXT NOT NULL,
                etf_name TEXT,
                date TEXT NOT NULL,
                stock_code TEXT,
                stock_name TEXT,
                isin TEXT,
                quantity INTEGER,
                weight_percent REAL,
                market_value REAL,
                current_price REAL,
                profit_loss REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date, stock_code)
            )
        """)
        
        logger.info("완전한 정보 저장용 테이블 생성 완료")
    
    def parse_filename(self, filename: str) -> Optional[Tuple[str, str, str]]:
        """
        파일명에서 ticker, date, file_type 추출
        
        Args:
            filename (str): 파일명 
            - 일반: "069500_2025-05-30_price.xls"
            - 월간: "069500_2025-04_distribution.xls" (월간 분배내역의 경우)
            
        Returns:
            Tuple[str, str, str]: (ticker, date, file_type) 또는 None
        """
        # 날짜는 YYYY-MM-DD 또는 YYYY-MM 형태 지원
        pattern = r"(\d+)_(\d{4}-\d{2}(?:-\d{2})?)_(price|distribution)\.xls"
        match = re.match(pattern, filename)
        
        if match:
            return match.group(1), match.group(2), match.group(3)
        
        logger.warning(f"XLS 파일명 패턴 매칭 실패: {filename}")
        return None

    def _get_etf_name(self, ticker: str) -> str:
        """ETF 매핑에서 ETF 이름을 가져오기 - 매핑이 없으면 ETF_티커 형태로 반환"""
        try:
            # etf_mapping.json 파일에서 ETF 이름 로드
            etf_name = get_etf_name(ticker)
            if etf_name:
                return etf_name
        except Exception as e:
            logger.debug(f"ETF 매핑 로드 실패: {e}")
        
        # 매핑에서 찾지 못한 경우 ETF_티커 형태로 반환 (하드코딩 제거)
        return f"ETF_{ticker}"
    
    def process_price_file(self, file_path: str, ticker: str, date: str) -> bool:
        """
        Price XLS 파일 처리 - 모든 정보를 완전히 저장
        
        주요 개선사항:
        - 모든 price.xls 컬럼 정보 저장
        - 3개월치 일별 데이터 개별 저장
        - 중복 데이터 자동 방지
        - ETF 이름 추가
        """
        try:
            # 이미 처리된 파일인지 확인
            if self.state_manager.is_xls_processed(file_path):
                logger.info(f"이미 처리된 가격 파일: {file_path}")
                return True
                
            logger.info(f"가격 파일 처리 시작: {file_path}")
            
            # Excel 파일 읽기
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                logger.error(f"Excel 파일 읽기 실패: {e}")
                return False
            
            if df.empty:
                logger.warning(f"빈 데이터프레임: {file_path}")
                return False
            
            # ETF 이름 가져오기
            etf_name = self._get_etf_name(ticker)
            
            # 가격 데이터 처리 - 모든 정보 추출
            try:
                records_inserted = 0
                
                with sqlite3.connect(self.db_full_path) as conn:
                    cursor = conn.cursor()
                    
                    # 헤더 행 찾기 - '일자'와 '거래가격' 또는 '종가'가 포함된 행
                    header_row = None
                    for i in range(min(5, len(df))):
                        row_str = ' '.join(str(val) for val in df.iloc[i].values if pd.notna(val))
                        if '일자' in row_str and ('거래가격' in row_str or '종가' in row_str):
                            header_row = i
                            break
                    
                    if header_row is None:
                        logger.warning(f"헤더 행을 찾을 수 없음: {file_path}")
                        return False
                    
                    # 실제 데이터 시작 행부터 처리 (헤더 다음 행부터)
                    data_start_row = header_row + 1
                    
                    for i in range(data_start_row, len(df)):
                        try:
                            row = df.iloc[i]
                            
                            # 날짜 추출 및 검증 (첫 번째 컬럼)
                            date_value = row.iloc[0]
                            if pd.isna(date_value):
                                continue
                                
                            # 날짜 형식 통일 (YYYYMMDD → YYYY-MM-DD)
                            try:
                                if isinstance(date_value, str):
                                    if len(date_value) == 8 and date_value.isdigit():
                                        # YYYYMMDD → YYYY-MM-DD
                                        date_str = f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:8]}"
                                    else:
                                        date_str = date_value.replace('/', '-')
                                else:
                                    date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
                            except:
                                continue
                            
                            # 모든 가격 정보 추출
                            closing_price = self._safe_float(row.iloc[1]) if len(row) > 1 else None
                            price_change = self._safe_float(row.iloc[2]) if len(row) > 2 else None
                            price_change_rate = self._safe_float(row.iloc[3]) if len(row) > 3 else None
                            volume = self._safe_int(row.iloc[4]) if len(row) > 4 else None
                            nav = self._safe_float(row.iloc[5]) if len(row) > 5 else None
                            nav_change = self._safe_float(row.iloc[6]) if len(row) > 6 else None
                            nav_change_rate = self._safe_float(row.iloc[7]) if len(row) > 7 else None
                            tax_base_price = self._safe_float(row.iloc[8]) if len(row) > 8 else None
                            
                            # 유효한 데이터가 있는 경우만 저장
                            if closing_price or nav:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO etf_prices 
                                    (ticker, etf_name, date, closing_price, price_change, price_change_rate,
                                     volume, nav, nav_change, nav_change_rate, tax_base_price)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    ticker, etf_name, date_str, closing_price, price_change, price_change_rate,
                                    volume, nav, nav_change, nav_change_rate, tax_base_price
                                ))
                                
                                records_inserted += 1
                                
                        except Exception as e:
                            logger.debug(f"행 처리 중 오류 (행 {i}): {e}")
                            continue
                    
                    conn.commit()
                    
                # 처리 완료 상태 저장
                self.state_manager.mark_xls_processed(file_path, ticker, date, "price")
                logger.info(f"가격 파일 처리 완료: {file_path}, 일별 레코드 수: {records_inserted}")
                return True
                
            except Exception as e:
                logger.error(f"가격 데이터 처리 중 오류: {e}")
                return False
                
        except Exception as e:
            logger.error(f"가격 파일 처리 실패 {file_path}: {e}")
            return False
    
    def process_distribution_file(self, file_path: str, ticker: str, date: str) -> bool:
        """
        Distribution XLS 파일 처리 - 모든 정보를 완전히 저장
        
        주요 개선사항:
        - 모든 distribution.xls 컬럼 정보 저장 
        - stock_name, isin, stock_code, quantity, weight_percent, market_value, current_price, profit_loss 모두 포함
        - ETF 이름 추가
        - ranking 제거 (ORDER BY로 처리 가능)
        """
        try:
            # 이미 처리된 파일인지 확인
            if self.state_manager.is_xls_processed(file_path):
                logger.info(f"이미 처리된 분배 파일: {file_path}")
                return True
                
            logger.info(f"분배 파일 처리 시작: {file_path}")
            
            # Excel 파일 읽기
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                logger.error(f"Excel 파일 읽기 실패: {e}")
                return False
            
            if df.empty:
                logger.warning(f"빈 데이터프레임: {file_path}")
                return False
            
            # ETF 이름 가져오기
            etf_name = self._get_etf_name(ticker)
            
            # 구성종목 데이터 처리 - 모든 정보 추출
            try:
                records_inserted = 0
                
                with sqlite3.connect(self.db_full_path) as conn:
                    cursor = conn.cursor()
                    
                    # 기존 동일 ticker+date 데이터 삭제 (중복 방지)
                    cursor.execute("""
                        DELETE FROM etf_distributions 
                        WHERE ticker = ? AND date = ?
                    """, (ticker, date))
                    
                    # 헤더 행 찾기 - '번호'와 '종목명'이 포함된 행
                    header_row = None
                    for i in range(min(5, len(df))):
                        row_str = ' '.join(str(val) for val in df.iloc[i].values if pd.notna(val))
                        if '번호' in row_str and '종목명' in row_str:
                            header_row = i
                            break
                    
                    if header_row is None:
                        logger.warning(f"헤더 행을 찾을 수 없음: {file_path}")
                        return False
                    
                    # 실제 데이터 시작 행부터 처리
                    data_start_row = header_row + 1
                    
                    for i in range(data_start_row, len(df)):
                        try:
                            row = df.iloc[i]
                            
                            # 모든 구성종목 정보 추출
                            # 컬럼 순서: 번호, 종목명, ISIN, 종목코드, 수량, 비중(%), 평가금액(원), 현재가(원), 등락(원)
                            
                            stock_name = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else None
                            
                            # '원화예금', '합계' 등은 제외 (실제 구성종목만 저장)
                            if not stock_name or stock_name in ['합계', '총계', '기타', 'nan']:
                                continue
                                
                            isin = str(row.iloc[2]).strip() if len(row) > 2 and pd.notna(row.iloc[2]) else None
                            stock_code = str(row.iloc[3]).strip() if len(row) > 3 and pd.notna(row.iloc[3]) else None
                            quantity = self._safe_int(row.iloc[4]) if len(row) > 4 else None
                            weight_percent = self._safe_float(row.iloc[5]) if len(row) > 5 else None
                            market_value = self._safe_float(row.iloc[6]) if len(row) > 6 else None
                            current_price = self._safe_float(row.iloc[7]) if len(row) > 7 else None
                            profit_loss = self._safe_float(row.iloc[8]) if len(row) > 8 else None
                            
                            # 유효한 구성종목 데이터가 있는 경우에만 삽입
                            if stock_name and (market_value or weight_percent):
                                cursor.execute("""
                                    INSERT OR REPLACE INTO etf_distributions 
                                    (ticker, etf_name, date, stock_code, stock_name, isin, quantity,
                                     weight_percent, market_value, current_price, profit_loss)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    ticker, etf_name, date, stock_code, stock_name, isin, quantity,
                                    weight_percent, market_value, current_price, profit_loss
                                ))
                                
                                records_inserted += 1
                        
                        except Exception as e:
                            logger.debug(f"행 처리 중 오류 (행 {i}): {e}")
                            continue
                    
                    conn.commit()
                    
                    # 처리 완료 상태 저장
                    self.state_manager.mark_xls_processed(file_path, ticker, date, "distribution")
                    logger.info(f"분배 파일 처리 완료: {file_path}, 구성종목 수: {records_inserted}")
                    return True
                    
            except Exception as e:
                logger.error(f"분배 데이터 처리 중 오류: {e}")
                return False
                
        except Exception as e:
            logger.error(f"분배 파일 처리 실패 {file_path}: {e}")
            return False
    
    def _safe_float(self, value) -> Optional[float]:
        """안전한 float 변환"""
        try:
            if pd.isna(value) or value == '' or value is None or str(value).strip() == '-':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _safe_int(self, value) -> Optional[int]:
        """안전한 int 변환"""
        try:
            if pd.isna(value) or value == '' or value is None or str(value).strip() == '-':
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def process_xls_directory(self, directory: str = "data/etf_raw", limit: int = None) -> Dict[str, int]:
        """
        디렉토리 내 모든 XLS 파일 처리
        
        Args:
            directory (str): 처리할 디렉토리 경로
            limit (int): 처리할 파일 수 제한
            
        Returns:
            Dict[str, int]: 처리 결과 통계
        """
        directory_path = Path(self.base_dir) / directory
        results = {"processed": 0, "failed": 0, "skipped": 0}
        
        # 모든 XLS 파일 찾기
        xls_files = []
        for ticker_dir in directory_path.iterdir():
            if ticker_dir.is_dir():
                for file in ticker_dir.glob("*.xls"):
                    if file.name.endswith(('_price.xls', '_distribution.xls')):
                        xls_files.append(file)
        
        # limit 적용
        if limit:
            xls_files = xls_files[:limit]
        
        logger.info(f"처리할 XLS 파일: {len(xls_files)}개")
        
        for file_path in xls_files:
            str_path = str(file_path)
            
            # 이미 처리된 파일 건너뛰기
            if self.state_manager.is_xls_processed(str_path):
                results["skipped"] += 1
                logger.info(f"이미 처리됨: {file_path.name}")
                continue
            
            # 파일명 파싱
            parsed = self.parse_filename(file_path.name)
            if not parsed:
                logger.warning(f"파일명 파싱 실패: {file_path.name}")
                results["failed"] += 1
                continue
            
            ticker, date, file_type = parsed
            
            # 파일 타입별 처리
            success = False
            if file_type == "price":
                success = self.process_price_file(str_path, ticker, date)
            elif file_type == "distribution":
                success = self.process_distribution_file(str_path, ticker, date)
            
            if success:
                results["processed"] += 1
            else:
                results["failed"] += 1
        
        logger.info(f"XLS 처리 완료: {results}")
        return results
    
    def is_already_processed(self, file_path: str) -> bool:
        """
        파일이 이미 처리되었는지 확인 (StateManager 사용)
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            bool: 처리 여부
        """
        return self.state_manager.is_xls_processed(file_path)
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환 (StateManager 사용)"""
        return self.state_manager.get_xls_processing_stats()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ETF XLS 파일 처리 - 완전한 정보 저장")
    parser.add_argument(
        "--directory",
        default="data/etf_raw",
        help="처리할 디렉토리 경로"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="처리할 파일 수 제한"
    )
    parser.add_argument(
        "--db-path",
        default="data/etf_database.sqlite",
        help="SQLite 데이터베이스 파일 경로"
    )
    
    args = parser.parse_args()
    
    processor = ETFXLSProcessor(args.db_path)
    results = processor.process_xls_directory(args.directory, args.limit)
    
    print("\n=== 처리 결과 ===")
    print(f"처리됨: {results['processed']}")
    print(f"실패: {results['failed']}")
    print(f"건너뜀: {results['skipped']}")
    
    print("\n=== 처리 통계 ===")
    stats = processor.get_processing_stats()
    for key, value in stats.items():
        print(f"{key}: {value}") 