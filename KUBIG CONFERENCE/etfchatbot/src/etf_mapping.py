"""
ETF 매핑 데이터 관리 모듈

MongoDB의 etf_db/etf_library 컬렉션에서 ETF 데이터를 가져와서
ticker와 etf_name 매핑 정보를 생성합니다.
"""

import os
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETFMappingManager:
    """ETF 티커와 이름 매핑 관리 클래스"""
    
    def __init__(self, mongo_host: str = "localhost", mongo_port: int = 27017):
        """
        Args:
            mongo_host: MongoDB 호스트
            mongo_port: MongoDB 포트
        """
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port
        self.client = None
        self.db = None
        self.collection = None
        
        # 기본 매핑 파일 경로
        self.mapping_file_path = "data/etf_mapping.json"
        
    def connect_to_mongodb(self) -> bool:
        """MongoDB에 연결"""
        if not PYMONGO_AVAILABLE:
            logger.error("pymongo가 설치되지 않았습니다. pip install pymongo")
            return False
            
        try:
            # 환경변수에서 인증 정보 가져오기
            host = os.environ.get('EC2_HOST') or self.mongo_host
            port = int(os.environ.get('EC2_PORT', '27017')) or self.mongo_port
            user = os.environ.get('DB_USER')
            password = os.environ.get('DB_PASSWORD')
            
            logger.info(f"MongoDB 연결 정보: {host}:{port}, User: {user}")
            
            if user and password:
                # 인증이 필요한 경우
                uri = f"mongodb://{user}:{password}@{host}:{port}/?authSource=admin&authMechanism=SCRAM-SHA-1"
                self.client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    socketTimeoutMS=30000,
                    retryWrites=True,
                    retryReads=True,
                    maxPoolSize=1,
                )
            else:
                # 인증이 없는 경우
                self.client = MongoClient(host, port)
            
            self.db = self.client['etf_db']
            self.collection = self.db['etf_library']
            
            # 연결 테스트
            self.client.admin.command('ping')
            logger.info(f"MongoDB 연결 성공: {host}:{port}")
            
            # 컬렉션 접근 테스트
            count = self.collection.count_documents({})
            logger.info(f"etf_library 컬렉션의 문서 수: {count}")
            
            return True
            
        except Exception as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            return False
    
    def fetch_etf_mapping_from_mongodb(self) -> Dict[str, str]:
        """
        MongoDB에서 ETF 매핑 데이터 가져오기
        
        Returns:
            Dict[ticker, etf_name] 형태의 매핑 딕셔너리
        """
        if self.collection is None:
            logger.error("MongoDB 연결이 필요합니다")
            return {}
            
        try:
            # etf_library 컬렉션에서 ticker와 etf_name 조회
            cursor = self.collection.find({}, {"ticker": 1, "etf_name": 1, "_id": 0})
            
            mapping = {}
            for doc in cursor:
                ticker = doc.get('ticker')
                etf_name = doc.get('etf_name')
                
                if ticker and etf_name:
                    mapping[ticker] = etf_name
                    
            logger.info(f"MongoDB에서 {len(mapping)}개의 ETF 매핑 데이터를 가져왔습니다")
            return mapping
            
        except Exception as e:
            logger.error(f"MongoDB에서 매핑 데이터 가져오기 실패: {e}")
            return {}
    
    def save_mapping_to_file(self, mapping: Dict[str, str], file_path: str = None) -> bool:
        """
        매핑 데이터를 JSON 파일로 저장
        
        Args:
            mapping: ETF 매핑 딕셔너리
            file_path: 저장할 파일 경로 (기본값: data/etf_mapping.json)
            
        Returns:
            저장 성공 여부
        """
        if not file_path:
            file_path = self.mapping_file_path
            
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
                
            logger.info(f"ETF 매핑 데이터를 {file_path}에 저장했습니다")
            return True
            
        except Exception as e:
            logger.error(f"매핑 파일 저장 실패: {e}")
            return False
    
    def load_mapping_from_file(self, file_path: str = None) -> Dict[str, str]:
        """
        JSON 파일에서 매핑 데이터 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            ETF 매핑 딕셔너리
        """
        if not file_path:
            file_path = self.mapping_file_path
            
        try:
            if not os.path.exists(file_path):
                logger.warning(f"매핑 파일이 존재하지 않습니다: {file_path}")
                return {}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                
            logger.info(f"{file_path}에서 {len(mapping)}개의 ETF 매핑 데이터를 로드했습니다")
            return mapping
            
        except Exception as e:
            logger.error(f"매핑 파일 로드 실패: {e}")
            return {}
    
    def get_etf_name(self, ticker: str) -> Optional[str]:
        """
        티커로 ETF 이름 조회
        
        Args:
            ticker: ETF 티커
            
        Returns:
            ETF 이름 (없으면 None)
        """
        mapping = self.load_mapping_from_file()
        return mapping.get(ticker)
    
    def update_mapping_from_mongodb(self) -> bool:
        """
        MongoDB에서 최신 매핑 데이터를 가져와서 파일 업데이트
        
        Returns:
            업데이트 성공 여부
        """
        if not self.connect_to_mongodb():
            return False
            
        try:
            # MongoDB에서 매핑 데이터 가져오기
            mapping = self.fetch_etf_mapping_from_mongodb()
            
            if not mapping:
                logger.warning("MongoDB에서 가져온 매핑 데이터가 없습니다")
                return False
            
            # 파일로 저장
            success = self.save_mapping_to_file(mapping)
            
            if success:
                logger.info("ETF 매핑 데이터 업데이트 완료")
                
                # 샘플 데이터 출력
                sample_tickers = list(mapping.keys())[:5]
                logger.info("샘플 매핑 데이터:")
                for ticker in sample_tickers:
                    logger.info(f"  {ticker}: {mapping[ticker]}")
                    
            return success
            
        finally:
            if self.client:
                self.client.close()
    
    def get_all_tickers(self) -> List[str]:
        """
        모든 ETF 티커 목록 반환
        
        Returns:
            ETF 티커 리스트
        """
        mapping = self.load_mapping_from_file()
        return list(mapping.keys())
    
    def create_reverse_mapping(self) -> Dict[str, str]:
        """
        역매핑 생성 (etf_name -> ticker)
        
        Returns:
            ETF 이름을 키로 하는 매핑 딕셔너리
        """
        mapping = self.load_mapping_from_file()
        return {name: ticker for ticker, name in mapping.items()}


def main():
    """CLI 실행용 메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ETF 매핑 데이터 관리")
    parser.add_argument(
        "--update-from-mongodb",
        action="store_true",
        help="MongoDB에서 최신 매핑 데이터 업데이트"
    )
    parser.add_argument(
        "--mongo-host",
        default="localhost",
        help="MongoDB 호스트 (기본값: localhost)"
    )
    parser.add_argument(
        "--mongo-port",
        type=int,
        default=27017,
        help="MongoDB 포트 (기본값: 27017)"
    )
    parser.add_argument(
        "--output-file",
        default="data/etf_mapping.json",
        help="출력 파일 경로 (기본값: data/etf_mapping.json)"
    )
    
    args = parser.parse_args()
    
    manager = ETFMappingManager(args.mongo_host, args.mongo_port)
    manager.mapping_file_path = args.output_file
    
    if args.update_from_mongodb:
        success = manager.update_mapping_from_mongodb()
        if success:
            logger.info("ETF 매핑 데이터 업데이트 성공")
        else:
            logger.error("ETF 매핑 데이터 업데이트 실패")
            exit(1)
    else:
        # 기존 매핑 파일 정보 출력
        mapping = manager.load_mapping_from_file()
        if mapping:
            logger.info(f"현재 매핑된 ETF 수: {len(mapping)}")
            sample_tickers = list(mapping.keys())[:5]
            logger.info("샘플 매핑 데이터:")
            for ticker in sample_tickers:
                logger.info(f"  {ticker}: {mapping[ticker]}")
        else:
            logger.info("매핑 파일이 없거나 비어있습니다. --update-from-mongodb 옵션을 사용하세요")


if __name__ == "__main__":
    main() 