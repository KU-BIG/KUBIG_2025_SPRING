import logging
import chromadb
import re
from datetime import datetime, timedelta
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from chromadb.config import Settings
from chatbot.prompts.tool_prompts import DATE_DETECTION_PROMPT
from chatbot.core.date_utils import get_date_utils

logger = logging.getLogger(__name__)


def load_embedding_model(model_name="text-embedding-3-small"):
    """OpenAI 임베딩 모델을 로드"""
    return OpenAIEmbeddings(model=model_name)


class ETFVectorRetriever:
    """ETF 벡터 검색기 - BM25 + 벡터 하이브리드 검색"""
    
    def __init__(self, vectordb_path: str = "data/vectordb"):
        self.vectordb_path = vectordb_path
        self.embeddings = load_embedding_model()
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
        self.client = chromadb.PersistentClient(
            path=vectordb_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
    
    def _get_available_dates(self, collection_name: str) -> List[str]:
        """컬렉션에서 사용 가능한 날짜 메타데이터 수집"""
        try:
            collection = self.client.get_collection(collection_name)
            # 모든 메타데이터에서 날짜 정보 추출
            result = collection.get()
            dates = set()
            
            if result["metadatas"]:
                for metadata in result["metadatas"]:
                    if metadata and "date" in metadata:
                        dates.add(metadata["date"])
            
            return sorted(list(dates))
        except Exception as e:
            logger.error(f"날짜 메타데이터 수집 실패: {e}")
            return []
    
    def _extract_date_filter(self, query: str, tickers: List[str]) -> Optional[List[str]]:
        """LLM이 자유롭게 검색할 날짜들을 결정하고 유효성 검증"""
        try:
            # DateUtils로 날짜 감지
            date_utils = get_date_utils()
            detected_dates = date_utils.detect_dates_from_query(query)
            
            if not detected_dates or detected_dates == ["all"]:
                return detected_dates
            
            # 실제 사용 가능한 날짜들과 비교하여 검증
            all_available_dates = set()
            
            if tickers:
                for ticker in tickers:
                    collection_name = f"etf_{ticker}"
                    dates = self._get_available_dates(collection_name)
                    all_available_dates.update(dates)
            else:
                # 모든 ETF 컬렉션에서 날짜 수집
                collections = self.client.list_collections()
                for col in collections:
                    if col.name.startswith("etf_"):
                        dates = self._get_available_dates(col.name)
                        all_available_dates.update(dates)
            
            if not all_available_dates:
                logger.info("사용 가능한 날짜 메타데이터가 없음")
                return detected_dates  # 그래도 LLM이 결정한 날짜 사용
            
            # DateUtils로 날짜 검증 및 조정
            valid_dates = date_utils.validate_and_adjust_dates(detected_dates, list(all_available_dates))
            
            if valid_dates:
                logger.info(f"최종 검증된 유효 날짜들: {valid_dates}")
                return valid_dates
            else:
                logger.info("LLM이 선택한 날짜 중 유효한 것이 없음")
                return None
                
        except Exception as e:
            logger.error(f"LLM 날짜 선택 실패: {e}")
            return None
    
    def _create_hybrid_retriever(self, vectorstore, collection_name: str, k: int = 20):
        """특정 컬렉션에 대한 하이브리드 retriever 생성 (Vector + BM25)"""
        try:
            # 벡터 검색 retriever
            chroma_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k*2, "lambda_mult": 0.7}
            )
            
            # BM25 retriever를 위한 문서 준비
            collection = self.client.get_collection(collection_name)
            result = collection.get()
            
            documents = []
            for i in range(len(result["documents"])):
                doc_content = result["documents"][i]
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                
                if len(str(doc_content).strip()) > 50:  # 충분한 내용이 있는 문서만
                    documents.append(Document(page_content=str(doc_content), metadata=metadata))
            
            if len(documents) > 0:
                # BM25 retriever 생성
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = k
                
                # 앙상블 retriever 생성
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[chroma_retriever, bm25_retriever],
                    weights=[0.85, 0.15]  # 벡터(85%) + BM25(15%)
                )
                
                logger.info(f"컬렉션 {collection_name}에 대한 하이브리드 retriever 생성 완료 (문서 수: {len(documents)})")
                return ensemble_retriever
            else:
                logger.warning(f"컬렉션 {collection_name}에 충분한 문서가 없어 벡터 검색만 사용")
                return chroma_retriever
                
        except Exception as e:
            logger.error(f"하이브리드 retriever 생성 실패: {e}")
            # 실패 시 기본 벡터 검색 사용
            return vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "lambda_mult": 0.4}
            )
    
    def _create_filtered_hybrid_retriever(self, vectorstore, collection_name: str, selected_dates: List[str], k: int = 20):
        """날짜 필터링이 적용된 하이브리드 retriever 생성 (Vector + BM25)"""
        try:
            # 1. 날짜 필터링된 문서들을 먼저 가져오기
            filter_dict = {"date": {"$in": selected_dates}}
            
            # 충분한 문서를 가져와서 BM25용 문서 풀 생성
            all_filtered_docs = vectorstore.similarity_search(
                "",  # 빈 쿼리로 모든 필터링된 문서 가져오기
                k=500,  # 충분히 많은 문서 수
                filter=filter_dict
            )
            
            if not all_filtered_docs:
                logger.warning(f"날짜 필터링된 문서가 없음: {selected_dates}")
                return None
            
            # 2. 벡터 검색 retriever (날짜 필터링 적용)
            def filtered_vector_search(query: str, k: int = k):
                return vectorstore.similarity_search(query, k=k*2, filter=filter_dict)
            
            # 3. BM25 retriever (필터링된 문서들로만 구성)
            filtered_documents = []
            for doc in all_filtered_docs:
                if len(str(doc.page_content).strip()) > 50:
                    filtered_documents.append(doc)
            
            if len(filtered_documents) < 10:  # 최소 문서 수 체크
                logger.warning(f"BM25용 문서가 부족함: {len(filtered_documents)}개")
                # 벡터 검색만 사용
                class FilteredVectorRetriever:
                    def get_relevant_documents(self, query: str):
                        return filtered_vector_search(query, k)
                return FilteredVectorRetriever()
            
            # 4. BM25 retriever 생성
            bm25_retriever = BM25Retriever.from_documents(filtered_documents)
            bm25_retriever.k = k
            
            # 5. 커스텀 앙상블 retriever 생성
            class FilteredEnsembleRetriever:
                def __init__(self, bm25_retriever, vector_search_func, weights=[0.85, 0.15]):
                    self.bm25_retriever = bm25_retriever
                    self.vector_search_func = vector_search_func
                    self.weights = weights
                
                def get_relevant_documents(self, query: str):
                    # Vector 검색 결과
                    vector_docs = self.vector_search_func(query, k)
                    
                    # BM25 검색 결과
                    bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                    
                    # 결과 조합 및 중복 제거
                    combined_docs = []
                    seen_contents = set()
                    
                    # Vector 결과 (85% 가중치)
                    for doc in vector_docs[:int(k * self.weights[0])]:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_contents:
                            seen_contents.add(content_hash)
                            combined_docs.append(doc)
                    
                    # BM25 결과 (15% 가중치)
                    for doc in bm25_docs[:int(k * self.weights[1])]:
                        content_hash = hash(doc.page_content)
                        if content_hash not in seen_contents:
                            seen_contents.add(content_hash)
                            combined_docs.append(doc)
                    
                    return combined_docs[:k]
            
            ensemble_retriever = FilteredEnsembleRetriever(
                bm25_retriever, 
                filtered_vector_search,
                weights=[0.85, 0.15]
            )
            
            logger.info(f"날짜 필터링된 하이브리드 retriever 생성 완료 - 필터링된 문서: {len(filtered_documents)}개")
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"필터링된 하이브리드 retriever 생성 실패: {e}")
            return None

    def _apply_date_filter(self, vectorstore, query: str, k: int, selected_dates: List[str]):
        """LLM이 선택한 날짜들로 문서 검색 - 하이브리드 검색 지원"""
        try:
            if not selected_dates:
                return []
            
            if "all" in selected_dates:
                # 전체 기간: 필터 없이 검색
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": k, "lambda_mult": 0.4} # 해당 값을 바꾸면 다양성-정확도 trade-off 조절 가능
                )
                return retriever.get_relevant_documents(query)
            
            # 1. 날짜 필터링된 하이브리드 retriever 시도
            collection_name = vectorstore._collection.name if hasattr(vectorstore, '_collection') else "unknown"
            
            hybrid_retriever = self._create_filtered_hybrid_retriever(
                vectorstore, collection_name, selected_dates, k
            )
            
            if hybrid_retriever:
                logger.info("날짜 필터링된 하이브리드 검색 사용")
                return hybrid_retriever.get_relevant_documents(query)
            
            # 2. 하이브리드 실패 시 기존 벡터 검색 사용
            logger.info("하이브리드 검색 실패 - 벡터 검색만 사용")
            filter_dict = {"date": {"$in": selected_dates}}
            logger.info(f"날짜 필터 적용: {filter_dict}")

            # 중복을 대비해 k의 2배만큼 문서를 한 번에 검색
            docs = vectorstore.similarity_search(
                query, 
                k=2*k,  # 중복 제거 후 k개를 보장하기 위해 넉넉하게 가져옴
                filter=filter_dict
            )
            
            if not docs:
                return []

            # 3. 가져온 전체 결과에서 중복 제거 (기존 의도 반영)
            unique_docs = []
            seen_contents = set()
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_docs.append(doc)
            
            logger.info(f"날짜 필터 검색 완료: {len(unique_docs)}개 고유 문서 발견 (상위 {k}개 반환)")
            
            # 4. 최종적으로 가장 관련성 높은 k개 문서 반환
            return unique_docs[:k]
                
        except Exception as e:
            logger.error(f"날짜 필터 적용 실패: {e}")
            return []
    
    def _search_collection(self, collection_name: str, query: str, dates: Optional[List[str]], k: int) -> List[Document]:
        """단일 컬렉션에서 BM25 + 벡터 하이브리드 검색"""
        try:
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            if dates and "all" not in dates:
                # 날짜 필터링 적용
                return self._apply_date_filter(vectorstore, query, k, dates)
            else:
                # 하이브리드 검색 (날짜 필터 없음)
                hybrid_retriever = self._create_hybrid_retriever(vectorstore, collection_name, k)
                return hybrid_retriever.get_relevant_documents(query)
                
        except Exception as e:
            logger.error(f"{collection_name} 검색 실패: {e}")
            return []
    
    def search(self, query: str, tickers: List[str], dates: Optional[List[str]] = None, 
               k: int = 10) -> List[Document]:
        """ETF 컬렉션에서 BM25 + 벡터 하이브리드 검색"""
        all_docs = []
        available_collections = [col.name for col in self.client.list_collections()]
        
        # 날짜 필터링 처리
        if dates:
            logger.info(f"날짜 필터링 적용: {dates}")
        
        # 감지된 ETF 컬렉션에서 하이브리드 검색
        if tickers and "all" not in tickers:
            # 특정 ETF들 검색
            for ticker in tickers:
                collection_name = f"etf_{ticker}"
                if collection_name in available_collections:
                    docs = self._search_collection(collection_name, query, dates, k)
                    all_docs.extend(docs)
                    logger.info(f"{collection_name}에서 {len(docs)}개 문서 검색")
        else:
            # 모든 ETF 검색 (tickers가 비어있거나 "all"인 경우)
            etf_collections = [col for col in available_collections if col.startswith("etf_")]
            for collection_name in etf_collections:
                docs_per_collection = k // len(etf_collections) if etf_collections else k
                docs = self._search_collection(collection_name, query, dates, docs_per_collection)
                all_docs.extend(docs)
        
        logger.info(f"총 검색된 문서 수: {len(all_docs)}")
        return all_docs

    def detect_dates(self, query: str, tickers: List[str] = None) -> Optional[List[str]]:
        """사용자 질문에서 날짜 감지 및 유효성 검증 (가장 가까운 날짜 찾기 포함)"""
        return self._extract_date_filter(query, tickers or []) 