"""
ETF 유틸리티 함수들

ETF 매핑 데이터를 로드하고 활용하는 유틸리티 함수들을 제공합니다.
AWS 인스턴스에서 챗봇 서빙 시 S3에서 매핑 파일을 다운로드하여 사용할 수 있습니다.
"""

import os
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


class ETFMappingLoader:
    """ETF 매핑 데이터 로더 클래스"""
    
    def __init__(self, local_path: str = "data/etf_mapping.json"):
        """
        Args:
            local_path: 로컬 매핑 파일 경로
        """
        self.local_path = local_path
        self._mapping_cache = None
        self._reverse_mapping_cache = None
        
    def download_mapping_from_s3(self, bucket: str = None, s3_key: str = None) -> bool:
        """
        S3에서 ETF 매핑 파일 다운로드
        
        Args:
            bucket: S3 버킷명 (기본값: 환경변수에서 가져옴)
            s3_key: S3 키 (기본값: etf_mapping/etf_mapping.json)
            
        Returns:
            다운로드 성공 여부
        """
        if not BOTO3_AVAILABLE:
            logger.warning("boto3가 설치되지 않았습니다. pip install boto3")
            return False
            
        try:
            if not bucket:
                bucket = os.environ.get('AWS_S3_BUCKET')
                if not bucket:
                    logger.error("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다")
                    return False
            
            if not s3_key:
                s3_key = 'etf_mapping/etf_mapping.json'
            
            # S3 클라이언트 생성
            s3_client = boto3.client('s3')
            
            # 로컬 디렉토리 생성
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            
            # S3에서 다운로드
            s3_client.download_file(bucket, s3_key, self.local_path)
            logger.info(f"ETF 매핑 파일을 S3에서 다운로드했습니다: s3://{bucket}/{s3_key}")
            
            # 캐시 초기화
            self._mapping_cache = None
            self._reverse_mapping_cache = None
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.info("S3에 ETF 매핑 파일이 없습니다 (최초 빌드일 수 있음)")
            elif error_code == 'NoSuchBucket':
                logger.info("S3 버킷이 존재하지 않습니다 (최초 빌드일 수 있음)")
            else:
                logger.warning(f"S3 다운로드 실패: {e}")
            return False
        except Exception as e:
            logger.warning(f"S3 다운로드 중 오류 발생 (최초 빌드일 수 있음): {e}")
            return False
    
    def load_mapping(self, force_reload: bool = False) -> Dict[str, str]:
        """
        ETF 매핑 데이터 로드 (캐시 사용)
        
        Args:
            force_reload: 강제 재로드 여부
            
        Returns:
            ETF 매핑 딕셔너리 {ticker: etf_name}
        """
        if self._mapping_cache is None or force_reload:
            try:
                # 로컬 파일이 없으면 S3에서 다운로드 시도
                if not os.path.exists(self.local_path):
                    logger.info("로컬 ETF 매핑 파일이 없습니다. S3에서 다운로드를 시도합니다.")
                    self.download_mapping_from_s3()
                
                if os.path.exists(self.local_path):
                    with open(self.local_path, 'r', encoding='utf-8') as f:
                        self._mapping_cache = json.load(f)
                    logger.info(f"ETF 매핑 데이터를 로드했습니다: {len(self._mapping_cache)}개")
                else:
                    logger.warning("ETF 매핑 파일을 찾을 수 없습니다")
                    self._mapping_cache = {}
                    
            except Exception as e:
                logger.error(f"ETF 매핑 파일 로드 실패: {e}")
                self._mapping_cache = {}
        
        return self._mapping_cache.copy()
    
    def get_etf_name(self, ticker: str) -> Optional[str]:
        """
        티커로 ETF 이름 조회
        
        Args:
            ticker: ETF 티커
            
        Returns:
            ETF 이름 (없으면 None)
        """
        mapping = self.load_mapping()
        return mapping.get(ticker)
    
    def get_ticker(self, etf_name: str) -> Optional[str]:
        """
        ETF 이름으로 티커 조회
        
        Args:
            etf_name: ETF 이름
            
        Returns:
            ETF 티커 (없으면 None)
        """
        if self._reverse_mapping_cache is None:
            mapping = self.load_mapping()
            self._reverse_mapping_cache = {name: ticker for ticker, name in mapping.items()}
        
        return self._reverse_mapping_cache.get(etf_name)
    
    def search_etf_by_name(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        ETF 이름으로 검색
        
        Args:
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            
        Returns:
            검색 결과 리스트 [{ticker, etf_name}]
        """
        mapping = self.load_mapping()
        query_lower = query.lower()
        
        results = []
        for ticker, etf_name in mapping.items():
            if query_lower in etf_name.lower():
                results.append({
                    'ticker': ticker,
                    'etf_name': etf_name
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_all_etfs(self) -> List[Dict[str, str]]:
        """
        모든 ETF 정보 반환
        
        Returns:
            ETF 정보 리스트 [{ticker, etf_name}]
        """
        mapping = self.load_mapping()
        return [
            {'ticker': ticker, 'etf_name': etf_name}
            for ticker, etf_name in mapping.items()
        ]
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """
        유효한 ETF 티커인지 확인
        
        Args:
            ticker: ETF 티커
            
        Returns:
            유효 여부
        """
        mapping = self.load_mapping()
        return ticker in mapping
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """
        매핑 데이터 통계 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        mapping = self.load_mapping()
        
        # ETF 이름별 카테고리 분석 (간단한 키워드 기반)
        categories = {
            'kodex': 0,
            'arirang': 0,
            'tiger': 0,
            'kbstar': 0,
            'hanaro': 0,
            'others': 0
        }
        
        for etf_name in mapping.values():
            etf_name_lower = etf_name.lower()
            categorized = False
            
            for category in categories:
                if category != 'others' and category in etf_name_lower:
                    categories[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                categories['others'] += 1
        
        return {
            'total_etfs': len(mapping),
            'categories': categories,
            'file_path': self.local_path,
            'file_exists': os.path.exists(self.local_path)
        }


# 전역 인스턴스 (싱글톤 패턴)
_etf_mapping_loader = None

def get_etf_mapping_loader() -> ETFMappingLoader:
    """
    ETF 매핑 로더 싱글톤 인스턴스 반환
    
    Returns:
        ETFMappingLoader 인스턴스
    """
    global _etf_mapping_loader
    if _etf_mapping_loader is None:
        _etf_mapping_loader = ETFMappingLoader()
    return _etf_mapping_loader


# 편의 함수들
def get_etf_name(ticker: str) -> Optional[str]:
    """티커로 ETF 이름 조회"""
    return get_etf_mapping_loader().get_etf_name(ticker)


def get_etf_ticker(etf_name: str) -> Optional[str]:
    """ETF 이름으로 티커 조회"""
    return get_etf_mapping_loader().get_ticker(etf_name)


def search_etf(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """ETF 검색"""
    return get_etf_mapping_loader().search_etf_by_name(query, limit)


def is_valid_etf_ticker(ticker: str) -> bool:
    """유효한 ETF 티커 확인"""
    return get_etf_mapping_loader().is_valid_ticker(ticker)


def update_etf_mapping_from_s3() -> bool:
    """S3에서 ETF 매핑 데이터 업데이트"""
    return get_etf_mapping_loader().download_mapping_from_s3()


if __name__ == "__main__":
    # 테스트 코드
    loader = ETFMappingLoader()
    
    # 매핑 데이터 로드
    mapping = loader.load_mapping()
    print(f"로드된 ETF 수: {len(mapping)}")
    
    # 샘플 데이터 출력
    sample_items = list(mapping.items())[:5]
    print("샘플 매핑 데이터:")
    for ticker, etf_name in sample_items:
        print(f"  {ticker}: {etf_name}")
    
    # 통계 정보
    stats = loader.get_mapping_stats()
    print(f"매핑 통계: {stats}")
    
    # 검색 테스트
    search_results = loader.search_etf_by_name("kodex")
    print(f"'kodex' 검색 결과: {len(search_results)}개")
    for result in search_results[:3]:
        print(f"  {result['ticker']}: {result['etf_name']}") 