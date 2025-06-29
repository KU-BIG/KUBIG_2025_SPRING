"""SQL 쿼리 실행 모듈"""
import logging
import sqlite3
import re
from datetime import datetime
from typing import List, Dict
from chatbot.core.date_utils import get_date_utils

logger = logging.getLogger(__name__)


class SQLRunner:
    """SQL 쿼리 실행기 - 날짜 자동 조정 및 결과 필터링 포함"""
    
    def __init__(self, db_path: str = "data/etf_database.sqlite"):
        self.db_path = db_path
    
    def execute_query(self, query: str) -> List[Dict]:
        """SQL 쿼리 실행 - distribution 데이터는 토큰 절약을 위해 필터링"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(query)
            
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            # 결과가 없는 경우 가장 가까운 날짜로 재시도
            if not results and self._contains_specific_date(query):
                logger.info("정확한 날짜 데이터가 없어 가장 가까운 날짜로 재검색합니다.")
                modified_query = self._modify_query_for_closest_date(query, conn)
                if modified_query and modified_query != query:
                    logger.info(f"수정된 쿼리로 재시도: {modified_query}")
                    cursor = conn.execute(modified_query)
                    for row in cursor.fetchall():
                        results.append(dict(zip(columns, row)))
            
            conn.close()
            logger.info(f"SQL 실행 완료: {len(results)}개 레코드")
            
            # distribution 테이블 결과가 많은 경우 필터링 (토큰 절약)
            if len(results) > 100 and self._is_distribution_query(query):
                filtered_results = self._filter_distribution_results(results)
                logger.info(f"Distribution 결과 필터링: {len(results)}개 → {len(filtered_results)}개 레코드")
                return filtered_results
            
            return results
            
        except Exception as e:
            logger.error(f"SQL 실행 실패: {e}")
            return []
    
    def _contains_specific_date(self, query: str) -> bool:
        """쿼리에 특정 날짜 조건이 있는지 확인 - 날짜 함수도 포함"""
        # YYYY-MM-DD 형태의 날짜 패턴 확인
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        # 날짜 함수 패턴 확인 (CURRENT_DATE, NOW(), TODAY() 등)
        date_function_pattern = r'(?i)(CURRENT_DATE|NOW\(\)|TODAY\(\)|CURDATE\(\))'
        
        return bool(re.search(date_pattern, query)) or bool(re.search(date_function_pattern, query))
    
    def _modify_query_for_closest_date(self, query: str, conn) -> str:
        """가장 가까운 날짜로 쿼리를 수정 - 날짜 함수도 처리"""
        try:
            # 쿼리에서 날짜 추출
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            dates_in_query = re.findall(date_pattern, query)
            
            # 날짜 함수 패턴 확인
            date_function_pattern = r'(?i)(CURRENT_DATE|NOW\(\)|TODAY\(\)|CURDATE\(\))'
            date_functions = re.findall(date_function_pattern, query)
            
            target_date = None
            date_replacement_needed = False
            
            if dates_in_query:
                target_date = dates_in_query[0]  # 첫 번째 날짜 사용
            elif date_functions:
                # 날짜 함수가 있는 경우 현재 날짜 사용
                target_date = datetime.now().strftime('%Y-%m-%d')
                date_replacement_needed = True
                logger.info(f"날짜 함수 {date_functions[0]} 감지 - 현재 날짜 {target_date}로 변환")
            
            if not target_date:
                return query
            
            # 테이블 이름 확인 (etf_prices 또는 etf_distributions)
            table_name = None
            if 'etf_prices' in query.lower():
                table_name = 'etf_prices'
            elif 'etf_distributions' in query.lower():
                table_name = 'etf_distributions'
            
            if not table_name:
                return query
            
            # 티커 정보도 추출 (더 정확한 검색을 위해)
            ticker_match = re.search(r"ticker.*?IN.*?\('([^']+)'\)", query, re.IGNORECASE)
            ticker_condition = ""
            if ticker_match:
                ticker = ticker_match.group(1)
                ticker_condition = f"AND ticker = '{ticker}'"
            
            # 가장 가까운 날짜 찾기 (개선된 로직)
            closest_date_query = f"""
            SELECT date, ABS(julianday(date) - julianday('{target_date}')) as date_diff
            FROM {table_name} 
            WHERE 1=1 {ticker_condition}
            ORDER BY date_diff ASC, date DESC
            LIMIT 1
            """
            
            logger.info(f"가장 가까운 날짜 검색 쿼리: {closest_date_query}")
            cursor = conn.execute(closest_date_query)
            closest_date_result = cursor.fetchone()
            
            if closest_date_result:
                closest_date = closest_date_result[0]
                date_diff = closest_date_result[1]
                logger.info(f"목표 날짜 {target_date} → 가장 가까운 날짜 {closest_date} ({date_diff:.1f}일 차이)")
                
                # 원본 쿼리의 날짜를 가장 가까운 날짜로 교체
                if date_replacement_needed:
                    # 날짜 함수를 실제 날짜로 교체
                    for date_func in date_functions:
                        modified_query = re.sub(
                            rf'(?i){re.escape(date_func)}', 
                            f"'{closest_date}'", 
                            query
                        )
                        query = modified_query
                else:
                    # 기존 날짜를 가장 가까운 날짜로 교체
                    modified_query = query.replace(target_date, closest_date)
                    query = modified_query
                
                return query
            
            return query
            
        except Exception as e:
            logger.error(f"가장 가까운 날짜 검색 실패: {e}")
            return query
    
    def _is_distribution_query(self, query: str) -> bool:
        """distribution 테이블을 대상으로 하는 쿼리인지 확인"""
        return "etf_distributions" in query.lower()
    
    def _filter_distribution_results(self, results: List[Dict]) -> List[Dict]:
        """Distribution 결과 필터링 - 날짜별 상위 10개 종목 또는 비중 1% 이상"""
        if not results:
            return results
        
        filtered = []
        
        # 날짜별로 그룹화
        by_date_ticker = {}
        for result in results:
            date = result.get('date', 'unknown')
            ticker = result.get('ticker', 'unknown')
            key = f"{date}_{ticker}"
            
            if key not in by_date_ticker:
                by_date_ticker[key] = []
            by_date_ticker[key].append(result)
        
        # 각 날짜-ticker 조합별로 필터링
        for key, records in by_date_ticker.items():
            # 1. 비중 1% 이상인 종목들
            high_weight_records = [
                r for r in records 
                if r.get('weight_percent', 0) >= 1.0
            ]
            
            # 2. 비중 기준 상위 10개 종목
            records_sorted = sorted(
                records, 
                key=lambda x: x.get('weight_percent', 0), 
                reverse=True
            )
            top_10_records = records_sorted[:10]
            
            # 두 조건 중 더 많은 것을 선택 (하지만 최대 15개로 제한)
            if len(high_weight_records) >= len(top_10_records):
                selected = high_weight_records[:15]  # 최대 15개
                filter_type = f"비중 1% 이상 {len(selected)}개"
            else:
                selected = top_10_records
                filter_type = f"상위 10개"
            
            filtered.extend(selected)
            
            # 로그 출력
            date_ticker = key.replace('_', ' - ')
            logger.info(f"{date_ticker}: {filter_type} 선택 ({len(records)}개 중)")
        
        return filtered 