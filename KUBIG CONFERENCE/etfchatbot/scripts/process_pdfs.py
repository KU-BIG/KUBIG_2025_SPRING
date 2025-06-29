import re
import sys
import time
import os
import logging
import psutil
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pathlib import Path
from dotenv import load_dotenv
from src.vectorstore import VectorStore
from src.parser import process_single_pdf
from src.parser import process_single_xls
from src.graphparser.state import GraphState
from src.utils.state_manager import StateManager
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain.schema import Document

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# StateManager 초기화
state_manager = StateManager()

def load_processed_states():
    """처리된 PDF 파일 목록을 로드합니다. (StateManager 사용)"""
    return state_manager.load_pdf_states()

def save_processed_states(processed_states):
    """처리된 PDF 파일 목록을 저장합니다. (StateManager 사용)"""
    state_manager.save_pdf_states(processed_states)

def is_etf_file(filename: str) -> bool:
    """ETF 상품 관련 파일인지 확인합니다. (ticker_date_type.extension 형식)"""
    # 일반 패턴: ticker_YYYY-MM-DD_type.extension (price 추가됨)
    pattern1 = r"\d+_\d{4}-\d{2}-\d{2}_(price|distribution|agreement|prospectus|monthly)\.(pdf|xls)"
    # 월간 보고서 패턴: ticker_YYYY-MM_monthly.pdf 
    pattern2 = r"\d+_\d{4}-\d{2}_monthly\.pdf"
    
    result = bool(re.match(pattern1, filename)) or bool(re.match(pattern2, filename))
    
    if not result:
        logger.debug(f"ETF 파일 패턴 매칭 실패: {filename}")
    else:
        logger.debug(f"ETF 파일 패턴 매칭 성공: {filename}")
    
    return result


def get_ticker_from_filename(filename: str) -> str:
    """파일명에서 ticker(상품 식별자)를 추출합니다."""
    parts = filename.split("_")
    if len(parts) >= 1:
        return parts[0]
    return ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception, ValueError)),
)
def process_single_pdf_with_retry(pdf_path):
    try:
        state = process_single_pdf(pdf_path)
        if state is None:
            raise ValueError(f"PDF 처리 실패: {pdf_path}")
        return state
    except Exception as e:
        logger.error(f"PDF 처리 재시도 중 오류: {str(e)}")
        raise


def process_single_pdf(pdf_path):
    try:
        logger.info(f"=== PDF 처리 시작: {pdf_path} ===")

        # PDF 파일 유효성 검사
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF 파일이 존재하지 않습니다: {pdf_path}")

        if os.path.getsize(pdf_path) == 0:
            raise ValueError(f"PDF 파일이 비어있습니다: {pdf_path}")

        # PDF 처리 시도
        try:
            from src.parser import process_single_pdf as parser_process_pdf

            state = parser_process_pdf(pdf_path)

            # 처리 결과 검증
            if state is None:
                raise ValueError(f"PDF 처리 결과가 없습니다: {pdf_path}")

            required_keys = [
                "text_summary",
                "image_summary",
                "table_summary",
            ]
            missing_keys = [key for key in required_keys if key not in state]
            if missing_keys:
                logger.warning(f"누락된 키가 있습니다: {missing_keys}")
                # 누락된 키에 대해 빈 딕셔너리 추가
                for key in missing_keys:
                    state[key] = {}

            logger.info(f"PDF 처리 완료: {os.path.basename(pdf_path)}")
            logger.info(f"처리된 데이터 키: {list(state.keys())}")
            return state

        except Exception as e:
            logger.error(f"PDF 파싱 중 오류 발생: {str(e)}", exc_info=True)
            # 기본 상태 반환
            return {
                "text_summary": {},
                "image_summary": {},
                "table_summary": {},
                "table_markdown": {},
            }

    except Exception as e:
        logger.error(f"PDF 처리 중 치명적 오류 발생: {str(e)}", exc_info=True)
        return None


def parse_etf_filename(filename: str) -> dict:
    """
    ETF 파일명에서 메타데이터 추출
    
    Args:
        filename (str): 파일명 
        - 일반 파일: "069500_2025-05-30_monthly.pdf"
        - 월간 보고서: "069500_2025-04_monthly.pdf"
        
    Returns:
        dict: 파싱된 메타데이터
    """
    # ETF 파일명 패턴: ticker_date_type.extension (price 추가됨)
    # 날짜는 YYYY-MM-DD 또는 YYYY-MM 형태 (월간 보고서의 경우)
    pattern = r"(\d+)_(\d{4}-\d{2}(?:-\d{2})?)_(price|distribution|agreement|prospectus|monthly)\.(pdf|xls)"
    match = re.match(pattern, filename)
    
    if not match:
        logger.warning(f"파일명 패턴 매칭 실패: {filename}")
        return {}
    
    ticker, date, doc_type, extension = match.groups()
    
    # ETF 이름 매핑 로드 (etf_mapping.json에서)
    try:
        from src.utils.etf_utils import get_etf_name
        etf_name = get_etf_name(ticker)
        if not etf_name:
            etf_name = f"etf_{ticker}"
    except Exception as e:
        logger.warning(f"ETF 이름 매핑 로드 실패: {e}")
        etf_name = f"etf_{ticker}"
    
    # 날짜가 이미 월 형태(YYYY-MM)인지 확인
    if len(date.split('-')) == 2:  # YYYY-MM 형태
        # 이미 월 형태이므로 그대로 사용
        final_date = date
    else:  # YYYY-MM-DD 형태
        if doc_type == "monthly":
            # 월간보고서는 월 단위로 변환
            date_parts = date.split("-")
            final_date = f"{date_parts[0]}-{date_parts[1]}"  # YYYY-MM 형태로 변환
        else:
            # 다른 문서는 그대로 사용
            final_date = date
    
    return {
        "ticker": ticker,
        "etf_name": etf_name,
        "doc_type": doc_type,
        "date": final_date,
        "extension": extension
    }


def get_collection_name_for_ticker(ticker: str) -> str:
    """
    ticker에 해당하는 ChromaDB 컬렉션 이름 반환
    
    Args:
        ticker (str): ETF 티커 (예: "069500")
        
    Returns:
        str: 컬렉션 이름
    """
    return f"etf_{ticker}"


def process_new_etf_files(limit: int = None):
    """ETF 관련 파일(PDF, XLS)을 ticker별 컬렉션으로 분리하여 처리합니다.

    Args:
        limit (int, optional): 처리할 ETF 상품(ticker)의 최대 개수. 기본값은 None으로 모든 상품 처리
    """
    etf_raw_directory = "./data/etf_raw"
    processed_states = load_processed_states()

    # 디버깅: 기존 상태 출력
    print("\n=== 기존 처리 상태 (StateManager) ===")
    print(f"처리된 파일 수: {len(processed_states)}")

    # 디렉토리 확인 및 생성
    raw_dir = Path(etf_raw_directory)
    if not raw_dir.exists():
        logger.warning(f"ETF 원본 디렉토리를 찾을 수 없습니다: {etf_raw_directory}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        return

    # ticker별 디렉토리에서 파일 수집
    all_files = []
    for ticker_dir in raw_dir.iterdir():
        if ticker_dir.is_dir():
            for file in ticker_dir.iterdir():
                if file.is_file() and is_etf_file(file.name):
                    all_files.append(file)

    # ETF 상품(ticker)별로 파일 그룹화
    product_files = {}
    for file_path in all_files:
        filename = file_path.name
        ticker = get_ticker_from_filename(filename)
        if ticker:
            if ticker not in product_files:
                product_files[ticker] = []
            product_files[ticker].append(file_path)

    # 각 상품별 파일을 처리 유형별로 정렬 (price → distribution → agreement → prospectus → monthly)
    for ticker, file_paths in product_files.items():
        file_paths.sort(key=lambda f: {
            'price': 0,
            'distribution': 1,
            'agreement': 2,
            'prospectus': 3,
            'monthly': 4
        }.get(f.name.split('_')[-1].split('.')[0], 99))

    # 처리할 상품 목록
    products_to_process = list(product_files.keys())

    # limit이 지정된 경우 처리할 상품 수 제한
    if limit is not None and limit > 0:
        products_to_process = products_to_process[:limit]
        logger.info(f"처리할 ETF 상품을 {limit}개로 제한합니다.")

    logger.info(f"\n=== 새로운 ETF 상품 정보 ===")
    logger.info(f"처리할 ETF 상품: {len(products_to_process)}개")
    logger.info(f"ETF 상품 목록: {products_to_process}")

    if not products_to_process:
        logger.info("처리할 ETF 상품이 없습니다.")
        return

    # VectorStore 초기화 (기본 컬렉션)
    logger.info("=== VectorStore 초기화 시작 ===")
    try:
        vector_store = VectorStore(persist_directory="./data/vectordb")
        logger.info("VectorStore 초기화 성공")
        
        # 기존 컬렉션 상태 확인
        try:
            collections = vector_store.client.list_collections()
            logger.info(f"기존 컬렉션 수: {len(collections)}")
            for col in collections:
                logger.info(f"  - {col.name}: {col.count()}개 문서")
        except Exception as e:
            logger.warning(f"기존 컬렉션 상태 확인 실패: {e}")
            
    except Exception as e:
        logger.error(f"VectorStore 초기화 실패: {e}")
        raise

    # 각 상품별 처리
    for ticker in products_to_process:
        logger.info(f"\n=== ETF 상품 처리 시작: {ticker} ===")
        file_paths = product_files.get(ticker, [])

        if not file_paths:
            logger.warning(f"티커 {ticker}에 대한 파일을 찾을 수 없습니다.")
            continue

        logger.info(f"처리할 파일 목록 ({len(file_paths)}개): {[f.name for f in file_paths]}")

        # ticker별 컬렉션 이름
        collection_name = get_collection_name_for_ticker(ticker)
        
        # 각 파일 순차 처리
        for file_path in file_paths:
            filename = file_path.name

            # 이미 처리된 파일이면 건너뛰기 (StateManager 사용)
            if state_manager.is_pdf_processed(filename, check_type="both"):
                logger.info(f"파일이 이미 처리되었습니다: {filename}")
                continue

            logger.info(f"\n--- 파일 처리 시작: {filename} ---")

            try:
                # 파일명에서 메타데이터 추출
                file_metadata = parse_etf_filename(filename)
                if not file_metadata:
                    logger.warning(f"파일명 파싱 실패: {filename}")
                    continue

                # PDF 파일만 벡터스토어에 저장 (XLS는 SQLite에서 처리)
                if filename.lower().endswith('.pdf'):
                    # Contextual Retrieval 사용 여부 설정
                    use_contextual_retrieval = True  # True로 변경하여 GPT 요약 기능 사용
                    
                    if use_contextual_retrieval:
                        # Agreement 파일인지 확인
                        if file_metadata.get("doc_type") == "agreement":
                            # Agreement 전용 향상된 청킹 방식 사용
                            logger.info(f"Agreement 파일 향상된 처리: {filename}")
                            
                            try:
                                # Agreement 전용 향상된 로딩 (서문 보존 + 조별 청킹 + Contextual)
                                documents = VectorStore.load_agreement_pdf_with_enhanced_chunking(
                                    filepath=str(file_path),
                                    max_tokens=1000,
                                    add_contextual_summary=True,
                                    document_metadata=file_metadata
                                )
                                
                                logger.info(f"Agreement 향상된 처리 완료: {len(documents)}개 청크 생성")
                                
                                # 처리 결과 통계
                                preamble_count = sum(1 for doc in documents if doc.metadata.get("chunk_type") == "preamble")
                                article_count = sum(1 for doc in documents if doc.metadata.get("chunk_type") == "article")
                                multi_part_count = sum(1 for doc in documents if doc.metadata.get("total_parts", 1) > 1)
                                
                                logger.info(f"  - 서문 청크: {preamble_count}개")
                                logger.info(f"  - 조별 청크: {article_count}개") 
                                logger.info(f"  할된 긴 조: {multi_part_count}개")
                                
                                # 상태 정보 업데이트
                                state_dict = {
                                    "agreement_preamble_chunks": preamble_count,
                                    "agreement_article_chunks": article_count,
                                    "agreement_multi_part_chunks": multi_part_count,
                                    "total_chunks": len(documents),
                                    "contextual_retrieval_processed": True,
                                    "vectorstore_processed": True,
                                    "metadata": file_metadata,
                                    "processing_method": "agreement_enhanced_chunking",
                                }

                                # Agreement 파일의 chunk 내용을 간단히 저장 (text_summary와 유사한 형태)
                                chunks_content = {}
                                for i, doc in enumerate(documents):
                                    chunk_key = f"chunk_{i}"
                                    chunks_content[chunk_key] = {
                                        "type": doc.metadata.get("chunk_type", "unknown"),
                                        "content": doc.page_content
                                    }
                                
                                state_dict["chunks_content"] = chunks_content
                                
                                # StateManager를 통해 상태 저장
                                state_manager.mark_pdf_processed(filename, state_dict)
                                
                                # 벡터스토어에 직접 저장 (이미 메타데이터 포함됨)
                                if documents:
                                    vector_store.add_documents(documents=documents, collection_name=collection_name)
                                    logger.info(f"Agreement 향상된 처리: {len(documents)}개 청크를 컬렉션 '{collection_name}'에 추가")
                                    
                                    # 샘플 청크 정보 출력 (처음 3개)
                                    for i, doc in enumerate(documents[:3]):
                                        chunk_type = doc.metadata.get("chunk_type", "unknown")
                                        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
                                        content_preview = doc.page_content[:100].replace('\n', ' ')
                                        logger.info(f"  샘플 청크 {i+1} ({chunk_type}): {chunk_id} - {content_preview}...")
                                
                                continue  # 다음 파일로 이동
                                
                            except Exception as e:
                                logger.error(f"Agreement 향상된 처리 실패: {e}", exc_info=True)
                                logger.info("기본 하이브리드 방식으로 fallback 처리합니다.")
                                # 에러 시 하이브리드 방식으로 처리 계속
                        else:
                            # 하이브리드 방식: 기존 처리 + 텍스트 Contextual 개선
                            logger.info(f"하이브리드 방식으로 처리: {filename}")
                            
                            try:
                                # 1단계: 기존 방식으로 이미지/테이블 처리
                                state = process_single_pdf_with_retry(str(file_path))
                                
                                if state is None:
                                    logger.error(f"PDF 파일 처리 실패: {filename}")
                                    continue

                                # 2단계: 텍스트 요약에 Contextual 정보 추가
                                enhanced_text_summary = {}
                                document_context = f"이 문서는 {file_metadata['etf_name']}({file_metadata['ticker']}) ETF의 {file_metadata['doc_type']} 문서({file_metadata['date']})입니다."
                                
                                for page, text in state.get("text_summary", {}).items():
                                    if text.strip():
                                        # GPT로 contextual 정보 추가
                                        from src.vectorstore import add_contextual_summary_to_chunk
                                        enhanced_text = add_contextual_summary_to_chunk(text, document_context)
                                        enhanced_text_summary[page] = enhanced_text
                                        logger.info(f"페이지 {page} 텍스트에 contextual 정보 추가")
                                
                                # 3단계: 상태 정보 업데이트 (기존 + 개선된 텍스트)
                                state_dict = {
                                    "text_summary": enhanced_text_summary,  # Contextual 정보가 추가된 텍스트
                                    "image_summary": state.get("image_summary", {}),  # 기존 이미지 분석 유지
                                    "table_summary": state.get("table_summary", {}),  # 기존 테이블 분석 유지
                                    "table_markdown": state.get("table_markdown", {}),  # 기존 테이블 마크다운 유지
                                    "contextual_retrieval_processed": True,
                                    "parsing_processed": True,
                                    "vectorstore_processed": True,
                                    "metadata": file_metadata,
                                    # 디버깅 정보
                                    "processing_method": "hybrid_contextual",
                                    "enhanced_text_pages": len(enhanced_text_summary),
                                }

                                # StateManager를 통해 상태 저장
                                state_manager.mark_pdf_processed(filename, state_dict)
                                
                                # 4단계: 처리 결과 요약
                                logger.info(f"\n=== 하이브리드 처리 완료: {filename} ===")
                                logger.info(f"향상된 텍스트 요약 수: {len(enhanced_text_summary)}")
                                logger.info(f"이미지 요약 수: {len(state_dict.get('image_summary', {}))}")
                                logger.info(f"테이블 요약 수: {len(state_dict.get('table_summary', {}))}")
                                logger.info(f"테이블 마크다운 수: {len(state_dict.get('table_markdown', {}))}")

                                # 5단계: 벡터스토어에 저장
                                documents = []

                                # 향상된 텍스트 요약 추가
                                for page, text in enhanced_text_summary.items():
                                    if text.strip():
                                        documents.append(
                                            Document(
                                                page_content=text,
                                                metadata={
                                                    "source": filename,
                                                    "type": "enhanced_text_summary", 
                                                    "page": page,
                                                    "etf_name": file_metadata["etf_name"],
                                                    "ticker": file_metadata["ticker"],
                                                    "doc_type": file_metadata["doc_type"],
                                                    "date": file_metadata["date"],
                                                },
                                            )
                                        )

                                # 이미지 요약 추가 (기존 방식 유지)
                                for page, text in state_dict.get("image_summary", {}).items():
                                    if text.strip():
                                        documents.append(
                                            Document(
                                                page_content=text,
                                                metadata={
                                                    "source": filename,
                                                    "type": "image_summary",
                                                    "page": page,
                                                    "etf_name": file_metadata["etf_name"],
                                                    "ticker": file_metadata["ticker"],
                                                    "doc_type": file_metadata["doc_type"],
                                                    "date": file_metadata["date"],
                                                },
                                            )
                                        )

                                # 테이블 요약 추가 (기존 방식 유지)
                                for page, text in state_dict.get("table_summary", {}).items():
                                    if text.strip():
                                        documents.append(
                                            Document(
                                                page_content=text,
                                                metadata={
                                                    "source": filename,
                                                    "type": "table_summary",
                                                    "page": page,
                                                    "etf_name": file_metadata["etf_name"],
                                                    "ticker": file_metadata["ticker"],
                                                    "doc_type": file_metadata["doc_type"],
                                                    "date": file_metadata["date"],
                                                },
                                            )
                                        )
                                
                                # 벡터스토어에 추가
                                if documents:
                                    vector_store.add_documents(documents=documents, collection_name=collection_name)
                                    logger.info(f"하이브리드 처리: {len(documents)}개 문서를 컬렉션 '{collection_name}'에 추가")
                                
                            except Exception as e:
                                logger.error(f"하이브리드 처리 실패: {e}", exc_info=True)
                                continue
                    else:
                        # 기존 방식: 페이지별 요약 사용
                        logger.info(f"기존 방식으로 처리: {filename}")
                        
                        state = process_single_pdf_with_retry(str(file_path))
                        
                        if state is None:
                            logger.error(f"PDF 파일 처리 실패: {filename}")
                            continue

                            # 상태 정보 업데이트
                            state_dict = {
                                "text_summary": state.get("text_summary", {}),
                                "image_summary": state.get("image_summary", {}),
                                "table_summary": state.get("table_summary", {}),
                                "table_markdown": state.get("table_markdown", {}),
                                "parsing_processed": True,
                                "vectorstore_processed": True,
                                "metadata": file_metadata,
                            }

                            # StateManager를 통해 상태 저장
                            state_manager.mark_pdf_processed(filename, state_dict)
                            logger.info(f"새로운 상태 추가됨: {filename}")

                            # 처리 결과 요약
                            logger.info(f"\n=== 처리 완료: {filename} ===")
                            logger.info(f"텍스트 요약 수: {len(state_dict['text_summary'])}")
                            logger.info(f"이미지 요약 수: {len(state_dict.get('image_summary', {}))}")
                            logger.info(f"테이블 요약 수: {len(state_dict.get('table_summary', {}))}")
                            logger.info(f"테이블 마크다운 수: {len(state_dict.get('table_markdown', {}))}")

                            # 텍스트 요약을 ticker별 컬렉션에 저장
                            documents = []

                            # 텍스트 요약 추출
                            for page, text in state_dict["text_summary"].items():
                                if text.strip():  # 빈 텍스트 제외
                                    documents.append(
                                        Document(
                                            page_content=text,
                                            metadata={
                                                "source": filename,
                                                "type": "text_summary", 
                                                "page": page,
                                                "etf_name": file_metadata["etf_name"],
                                                "ticker": file_metadata["ticker"],
                                                "doc_type": file_metadata["doc_type"],
                                                "date": file_metadata["date"],
                                            },
                                        )
                                    )

                            # 이미지 요약도 추가 (있는 경우)
                            for page, text in state_dict.get("image_summary", {}).items():
                                if text.strip():
                                    documents.append(
                                        Document(
                                            page_content=text,
                                            metadata={
                                                "source": filename,
                                                "type": "image_summary",
                                                "page": page,
                                                "etf_name": file_metadata["etf_name"],
                                                "ticker": file_metadata["ticker"],
                                                "doc_type": file_metadata["doc_type"],
                                                "date": file_metadata["date"],
                                            },
                                        )
                                    )

                            # 테이블 요약도 추가 (있는 경우)
                            for page, text in state_dict.get("table_summary", {}).items():
                                if text.strip():
                                    documents.append(
                                        Document(
                                            page_content=text,
                                            metadata={
                                                "source": filename,
                                                "type": "table_summary",
                                                "page": page,
                                                "etf_name": file_metadata["etf_name"],
                                                "ticker": file_metadata["ticker"],
                                                "doc_type": file_metadata["doc_type"],
                                                "date": file_metadata["date"],
                                            },
                                        )
                                    )

                            # 문서가 있는 경우에만 ticker별 컬렉션에 추가
                            if documents:
                                logger.info(f"=== ChromaDB 컬렉션 '{collection_name}'에 문서 추가 시작 ===")
                                logger.info(f"추가할 문서 수: {len(documents)}")
                                for i, doc in enumerate(documents[:3]):  # 처음 3개만 로깅
                                    logger.info(f"문서 {i+1}: {doc.page_content[:100]}...")
                                    logger.info(f"메타데이터: {doc.metadata}")
                                
                                try:
                                    vector_store.add_documents(documents=documents, collection_name=collection_name)
                                    logger.info(f"{len(documents)}개 문서를 컬렉션 '{collection_name}'에 성공적으로 추가했습니다.")
                                    
                                    # 추가 후 컬렉션 상태 확인
                                    try:
                                        collection = vector_store.client.get_collection(collection_name)
                                        total_count = collection.count()
                                        logger.info(f"컬렉션 '{collection_name}' 현재 총 문서 수: {total_count}")
                                    except Exception as verify_error:
                                        logger.warning(f"컬렉션 상태 확인 실패: {verify_error}")
                                        
                                except Exception as add_error:
                                    logger.error(f"문서 추가 실패: {add_error}", exc_info=True)
                                    continue
                            else:
                                logger.warning(f"추가할 문서가 없습니다: {filename}")
                        
                elif filename.lower().endswith('.xls'):
                    # XLS 파일은 별도로 처리되므로 여기서는 상태만 기록
                    logger.info(f"XLS 파일은 별도 처리됩니다: {filename}")
                    state_dict = {
                        "parsing_processed": True,
                        "vectorstore_processed": False,  # XLS는 벡터스토어에 저장하지 않음
                        "metadata": file_metadata,
                    }
                    state_manager.mark_pdf_processed(filename, state_dict)

            except Exception as e:
                logger.error(f"처리 실패 ({filename}): {str(e)}", exc_info=True)
                continue

        logger.info(f"\n=== ETF 상품 처리 완료: {ticker} ===")

    logger.info(f"\n=== 전체 처리 완료: {len(products_to_process)}개 ETF 상품 ===")
    
    # 최종 컬렉션 상태 요약
    logger.info("\n=== ChromaDB 컬렉션 상태 요약 ===")
    for ticker in products_to_process:
        collection_name = get_collection_name_for_ticker(ticker)
        try:
            collection = vector_store.client.get_collection(collection_name)
            doc_count = collection.count()
            logger.info(f"컬렉션 '{collection_name}': {doc_count}개 문서")
        except Exception as e:
            logger.info(f"컬렉션 '{collection_name}': 생성되지 않음 또는 오류 ({e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF 문서 처리 스크립트")
    parser.add_argument("--limit", type=int, help="처리할 ETF 상품 최대 개수")
    args = parser.parse_args()

    # ETF 파일 처리
    process_new_etf_files(limit=args.limit)
