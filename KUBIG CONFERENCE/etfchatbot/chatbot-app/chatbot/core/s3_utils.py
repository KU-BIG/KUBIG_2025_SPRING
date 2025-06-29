import logging
import subprocess
import chromadb
import os
from chromadb.config import Settings

logger = logging.getLogger(__name__)


def setup_aws_credentials():
    """AWS credentials 설정 (환경변수 또는 기본값 사용)"""
    try:
        # Jenkins 환경변수에서 AWS credentials 가져오기
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'ap-northeast-3')
        
        if aws_access_key and aws_secret_key:
            logger.info("환경변수에서 AWS credentials 발견, 설정 중...")
            
            # AWS CLI 설정
            os.system(f"aws configure set aws_access_key_id {aws_access_key}")
            os.system(f"aws configure set aws_secret_access_key {aws_secret_key}")
            os.system(f"aws configure set region {aws_region}")
            
            logger.info("AWS credentials 설정 완료")
            return True
        else:
            logger.warning("AWS credentials 환경변수를 찾을 수 없음")
            logger.info("로컬 실행 모드 또는 기존 AWS 설정 사용")
            return False
            
    except Exception as e:
        logger.error(f"AWS credentials 설정 실패: {str(e)}")
        return False


def download_all_from_s3():
    """S3에서 모든 필요한 데이터 다운로드 (VectorDB + ETF DB + 매핑 파일)"""
    try:
        logger.info("=== S3에서 전체 데이터 다운로드 시작 ===")
        
        # AWS credentials 설정 시도
        setup_aws_credentials()
        
        # 1. VectorDB 다운로드
        vectordb_success = download_vectordb_from_s3()
        
        # 2. ETF 데이터베이스 다운로드
        etf_db_success = download_etf_database_from_s3()
        
        # 3. ETF 매핑 파일 다운로드
        mapping_success = download_etf_mapping_from_s3()
        
        logger.info(f"=== S3 다운로드 완료 ===")
        logger.info(f"VectorDB: {'성공' if vectordb_success else '실패/생략'}")
        logger.info(f"ETF DB: {'성공' if etf_db_success else '실패/생략'}")
        logger.info(f"ETF Mapping: {'성공' if mapping_success else '실패/생략'}")
        
        return True
        
    except Exception as e:
        logger.error(f"S3 전체 다운로드 실패: {str(e)}")
        return False


def download_vectordb_from_s3(bucket_name=None):
    """S3에서 VectorDB 다운로드"""
    try:
        local_db_path = "data/vectordb"
        os.makedirs(local_db_path, exist_ok=True)
        
        # 버킷 이름을 환경변수 또는 인자에서 가져오기
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            if bucket_name is None:
                logger.error("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다")
                return False
        
        logger.info("=== VectorDB 다운로드 시작 ===")
        
        download_command = f"aws s3 sync s3://{bucket_name}/vectordb/ {local_db_path}"
        logger.info(f"VectorDB 다운로드 명령어: {download_command}")
        
        result = subprocess.run(
            download_command, shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("VectorDB 다운로드 성공")
            return True
        else:
            logger.warning(f"VectorDB 다운로드 경고: {result.stderr}")
            logger.info("초기 실행이거나 새로운 VectorDB 생성 예정")
            return False
            
    except Exception as e:
        logger.error(f"VectorDB 다운로드 실패: {str(e)}")
        return False


def download_etf_database_from_s3(bucket_name=None):
    """S3에서 ETF 데이터베이스 다운로드"""
    try:
        local_file = "data/etf_database.sqlite"
        os.makedirs("data", exist_ok=True)
        
        # 버킷 이름을 환경변수 또는 인자에서 가져오기
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            if bucket_name is None:
                logger.error("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다")
                return False
        
        logger.info("=== ETF 데이터베이스 다운로드 시작 ===")
        
        download_command = f"aws s3 cp s3://{bucket_name}/etf_database/etf_database.sqlite {local_file}"
        logger.info(f"ETF DB 다운로드 명령어: {download_command}")
        
        result = subprocess.run(
            download_command, shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("ETF 데이터베이스 다운로드 성공")
            if os.path.exists(local_file):
                size = os.path.getsize(local_file) / (1024 * 1024)  # MB
                logger.info(f"ETF DB 파일 크기: {size:.2f} MB")
            return True
        else:
            logger.warning(f"ETF 데이터베이스 다운로드 경고: {result.stderr}")
            logger.info("기존 ETF 데이터베이스가 없거나 새로 생성 예정")
            return False
            
    except Exception as e:
        logger.error(f"ETF 데이터베이스 다운로드 실패: {str(e)}")
        return False


def download_etf_mapping_from_s3(bucket_name=None):
    """S3에서 ETF 매핑 파일 다운로드"""
    try:
        local_file = "data/etf_mapping.json"
        os.makedirs("data", exist_ok=True)
        
        # 버킷 이름을 환경변수 또는 인자에서 가져오기
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            if bucket_name is None:
                logger.error("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다")
                return False
        
        logger.info("=== ETF 매핑 파일 다운로드 시작 ===")
        
        download_command = f"aws s3 cp s3://{bucket_name}/etf_mapping/etf_mapping.json {local_file}"
        logger.info(f"ETF 매핑 다운로드 명령어: {download_command}")
        
        result = subprocess.run(
            download_command, shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("ETF 매핑 파일 다운로드 성공")
            if os.path.exists(local_file):
                size = os.path.getsize(local_file) / 1024  # KB
                logger.info(f"ETF 매핑 파일 크기: {size:.2f} KB")
            return True
        else:
            logger.warning(f"ETF 매핑 파일 다운로드 경고: {result.stderr}")
            logger.info("기존 ETF 매핑 파일이 없거나 새로 생성 예정")
            return False
            
    except Exception as e:
        logger.error(f"ETF 매핑 파일 다운로드 실패: {str(e)}")
        return False


def load_vectorstore_from_s3():
    """S3에서 ChromaDB 데이터를 다운로드하고 클라이언트를 초기화 (기존 함수 - 호환성 유지)"""
    try:
        # 전체 데이터 다운로드
        download_all_from_s3()
        
        # ChromaDB 클라이언트 생성
        client = setup_chromadb_client("data/vectordb")
        
        logger.info("전체 데이터 로드 완료")
        return client
        
    except Exception as e:
        logger.error(f"S3에서 전체 데이터 로드 실패: {str(e)}")
        # 실패해도 로컬 클라이언트는 생성
        return setup_chromadb_client("data/vectordb")


def setup_chromadb_client(vectordb_path: str = "data/vectordb"):
    """ChromaDB 클라이언트 설정"""
    try:
        os.makedirs(vectordb_path, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=vectordb_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        logger.info(f"ChromaDB 클라이언트 초기화 완료: {vectordb_path}")
        
        # 사용 가능한 컬렉션 목록 출력
        collections = client.list_collections()
        logger.info(f"사용 가능한 컬렉션 수: {len(collections)}")
        for col in collections[:5]:  # 처음 5개만 출력
            logger.info(f"  - {col.name}")
        
        return client
        
    except Exception as e:
        logger.error(f"ChromaDB 클라이언트 설정 실패: {str(e)}")
        raise


def save_all_to_s3():
    """모든 데이터를 S3에 업로드 (VectorDB + ETF DB + 매핑 파일)"""
    try:
        logger.info("=== S3에 전체 데이터 업로드 시작 ===")
        
        # 1. VectorDB 업로드
        vectordb_success = save_vectordb_to_s3()
        
        # 2. ETF 데이터베이스 업로드  
        etf_db_success = save_etf_database_to_s3()
        
        # 3. ETF 매핑 파일 업로드
        mapping_success = save_etf_mapping_to_s3()
        
        logger.info(f"=== S3 업로드 완료 ===")
        logger.info(f"VectorDB: {'성공' if vectordb_success else '실패/생략'}")
        logger.info(f"ETF DB: {'성공' if etf_db_success else '실패/생략'}")
        logger.info(f"ETF Mapping: {'성공' if mapping_success else '실패/생략'}")
        
        return vectordb_success and etf_db_success and mapping_success
        
    except Exception as e:
        logger.error(f"S3 전체 업로드 실패: {str(e)}")
        return False


def save_vectordb_to_s3(bucket_name=None):
    """VectorDB를 S3에 업로드"""
    try:
        local_db_path = "data/vectordb"
        
        if not os.path.exists(local_db_path):
            logger.warning("VectorDB 디렉토리가 존재하지 않음")
            return False

        # 버킷 이름을 환경변수 또는 인자에서 가져오기
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            if bucket_name is None:
                logger.error("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다")
                return False

        logger.info("=== VectorDB S3 업로드 시작 ===")
        
        # 동기화 명령 실행
        sync_command = f"aws s3 sync {local_db_path} s3://{bucket_name}/vectordb --delete"
        logger.info(f"VectorDB 업로드 명령어: {sync_command}")

        result = subprocess.run(
            sync_command, shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("VectorDB S3 업로드 성공")
            return True
        else:
            logger.error(f"VectorDB S3 업로드 실패: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"VectorDB S3 업로드 실패: {str(e)}")
        return False


def save_etf_database_to_s3(bucket_name=None):
    """ETF 데이터베이스를 S3에 업로드"""
    try:
        local_file = "data/etf_database.sqlite"
        
        if not os.path.exists(local_file):
            logger.warning("ETF 데이터베이스 파일이 존재하지 않음")
            return False

        # 버킷 이름을 환경변수 또는 인자에서 가져오기
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            if bucket_name is None:
                logger.error("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다")
                return False

        logger.info("=== ETF 데이터베이스 S3 업로드 시작 ===")
        
        size = os.path.getsize(local_file) / (1024 * 1024)  # MB
        logger.info(f"ETF DB 파일 크기: {size:.2f} MB")
        
        upload_command = f"aws s3 cp {local_file} s3://{bucket_name}/etf_database/etf_database.sqlite"
        logger.info(f"ETF DB 업로드 명령어: {upload_command}")

        result = subprocess.run(
            upload_command, shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("ETF 데이터베이스 S3 업로드 성공")
            return True
        else:
            logger.error(f"ETF 데이터베이스 S3 업로드 실패: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"ETF 데이터베이스 S3 업로드 실패: {str(e)}")
        return False


def save_etf_mapping_to_s3(bucket_name=None):
    """ETF 매핑 파일을 S3에 업로드"""
    try:
        local_file = "data/etf_mapping.json"
        
        if not os.path.exists(local_file):
            logger.warning("ETF 매핑 파일이 존재하지 않음")
            return False

        # 버킷 이름을 환경변수 또는 인자에서 가져오기
        if bucket_name is None:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            if bucket_name is None:
                logger.error("AWS_S3_BUCKET 환경변수가 설정되지 않았습니다")
                return False

        logger.info("=== ETF 매핑 파일 S3 업로드 시작 ===")
        
        size = os.path.getsize(local_file) / 1024  # KB
        logger.info(f"ETF 매핑 파일 크기: {size:.2f} KB")
        
        upload_command = f"aws s3 cp {local_file} s3://{bucket_name}/etf_mapping/etf_mapping.json"
        logger.info(f"ETF 매핑 업로드 명령어: {upload_command}")

        result = subprocess.run(
            upload_command, shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("ETF 매핑 파일 S3 업로드 성공")
            return True
        else:
            logger.error(f"ETF 매핑 파일 S3 업로드 실패: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"ETF 매핑 파일 S3 업로드 실패: {str(e)}")
        return False


# 기존 함수들 (하위 호환성)
def save_to_s3():
    """ChromaDB 데이터를 S3에 업로드 (기존 함수 - VectorDB만)"""
    return save_vectordb_to_s3() 