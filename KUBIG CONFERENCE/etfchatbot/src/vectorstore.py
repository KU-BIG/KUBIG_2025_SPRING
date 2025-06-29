from dotenv import load_dotenv
import os
import json
import chromadb
from tqdm import tqdm
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
import logging
import re

logger = logging.getLogger(__name__)

def add_contextual_summary_to_chunk(chunk_text: str, document_context: str = "") -> str:
    """
    Anthropic의 Contextual Retrieval 방식으로 청크에 GPT 생성 요약 추가
    
    Args:
        chunk_text (str): 원본 청크 텍스트
        document_context (str): 문서 전체 컨텍스트 (파일명, 메타데이터 등)
    
    Returns:
        str: 요약이 추가된 청크 텍스트
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Anthropic의 프롬프트를 한국어로 적용
        prompt = f"""<document_context>
{document_context}
</document_context>

다음은 위 문서에서 추출한 텍스트 청크입니다:
<chunk>
{chunk_text}
</chunk>

이 청크를 전체 문서 맥락에서 이해할 수 있도록 간결한 맥락 정보를 제공해주세요. 검색 시 이 청크를 더 잘 찾을 수 있도록 도움이 되는 맥락만 간단히 작성하세요. 맥락 정보만 답변하고 다른 설명은 하지 마세요."""

        try:
            response = llm.invoke(prompt)
            context_summary = response.content.strip()
            
            # 요약이 너무 길면 자르기 (100토큰 정도로 제한)
            if len(context_summary) > 200:
                context_summary = context_summary[:200] + "..."
            
            # 맥락 요약 + 원본 청크 결합
            return f"{context_summary} {chunk_text}"
            
        except Exception as e:
            logger.warning(f"GPT 요약 생성 실패, 원본 텍스트 사용: {e}")
            return chunk_text
            
    except Exception as e:
        logger.error(f"Contextual summary 생성 중 오류: {e}")
        return chunk_text

def extract_agreement_with_preamble_and_articles(text: str) -> List[Dict[str, Any]]:
    """
    Agreement PDF를 서문 + 조별로 정확하게 분할
    
    Args:
        text (str): 전체 텍스트
        
    Returns:
        List[Dict]: [{"chunk_type": "preamble", "content": "..."}, {"chunk_type": "article", "chapter": "1", "article": "2", ...}]
    """
    chunks = []
    lines = text.split('\n')
    
    # 서문 수집 (제1장 이전)
    preamble_content = []
    current_chapter = None
    current_article = None
    current_content = []
    
    # 패턴 정의
    chapter_pattern = r'^\s*제\s*(\d+)\s*장\s*(.*)$'
    article_pattern = r'^\s*제\s*(\d+)\s*조\s*\(([^)]+)\)'
    
    for line_num, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # 제X장 패턴 검사
        chapter_match = re.match(chapter_pattern, line_stripped)
        if chapter_match:
            # 서문이 있고 아직 저장하지 않았다면 저장
            if not current_chapter and preamble_content:
                preamble_text = '\n'.join(preamble_content).strip()
                if preamble_text:
                    chunks.append({
                        "chunk_type": "preamble",
                        "content": preamble_text,
                        "chunk_id": "서문",
                        "chapter": "0",
                        "article": "0",
                        "chapter_title": "서문",
                        "article_title": "문서 정보"
                    })
            
            # 이전 조 저장
            if current_chapter and current_article and current_content:
                content = '\n'.join(current_content).strip()
                
                chunks.append({
                    "chunk_type": "article",
                    "chapter": current_chapter["num"],
                    "article": current_article["num"],
                    "chapter_title": current_chapter["title"],
                    "article_title": current_article["title"],
                    "content": content,
                    "chunk_id": f"제{current_chapter['num']}장제{current_article['num']}조"
                })
            
            # 새 장 시작
            current_chapter = {
                "num": chapter_match.group(1),
                "title": chapter_match.group(2).strip()
            }
            current_article = None
            current_content = []
            continue
        
        # 제X조 패턴 검사
        article_match = re.match(article_pattern, line_stripped)
        if article_match:
            # 이전 조 저장
            if current_chapter and current_article and current_content:
                content = '\n'.join(current_content).strip()
                
                chunks.append({
                    "chunk_type": "article",
                    "chapter": current_chapter["num"],
                    "article": current_article["num"],
                    "chapter_title": current_chapter["title"],
                    "article_title": current_article["title"],
                    "content": content,
                    "chunk_id": f"제{current_chapter['num']}장제{current_article['num']}조"
                })
            
            # 새 조 시작
            current_article = {
                "num": article_match.group(1),
                "title": article_match.group(2).strip()
            }
            current_content = [line_stripped]
            continue
        
        # 일반 내용 추가
        if current_chapter and current_article:
            current_content.append(line_stripped)
        elif not current_chapter:
            # 아직 장이 시작되지 않았으면 서문에 추가
            preamble_content.append(line_stripped)
    
    # 마지막 조 저장
    if current_chapter and current_article and current_content:
        content = '\n'.join(current_content).strip()
       
        chunks.append({
            "chunk_type": "article",
            "chapter": current_chapter["num"],
            "article": current_article["num"],
            "chapter_title": current_chapter["title"],
            "article_title": current_article["title"],
            "content": content,
            "chunk_id": f"제{current_chapter['num']}장제{current_article['num']}조"
        })
    
    # 서문이 아직 저장되지 않았다면 저장 (장이 하나도 없는 경우)
    if not current_chapter and preamble_content:
        preamble_text = '\n'.join(preamble_content).strip()
        if preamble_text:
            chunks.append({
                "chunk_type": "preamble",
                "content": preamble_text,
                "chunk_id": "서문",
                "chapter": "0",
                "article": "0",
                "chapter_title": "서문",
                "article_title": "문서 정보"
            })
    
    logger.info(f"Agreement 분할 완료: 서문 포함 {len(chunks)}개 청크 추출")
    return chunks

def split_long_articles(article_chunks: List[Dict[str, Any]], max_tokens: int = 1000) -> List[Dict[str, Any]]:
    """
    긴 조를 내부에서 분할 (조 간에는 overlap 없음, 조 내부에서만 overlap 적용)
    
    Args:
        article_chunks: 조별로 분할된 청크들
        max_tokens: 최대 토큰 수
        
    Returns:
        List[Dict]: 분할 완료된 청크들
    """
    final_chunks = []
    
    for chunk in article_chunks:
        content_length = len(chunk["content"])
        
        if content_length > max_tokens:
            # 조 내부에서만 overlap 적용하여 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_tokens,
                chunk_overlap=100,  # 조 내부에서만 overlap
                length_function=len
            )
            
            sub_chunks = text_splitter.split_text(chunk["content"])
            
            for i, sub_chunk in enumerate(sub_chunks):
                # 조 정보를 청크 앞에 추가
                if chunk["chunk_type"] == "preamble":
                    enhanced_content = f"[서문] {sub_chunk}"
                    chunk_id = f"서문_part{i+1}"
                else:
                    enhanced_content = f"[제{chunk['chapter']}장 제{chunk['article']}조 ({chunk['article_title']})] {sub_chunk}"
                    chunk_id = f"제{chunk['chapter']}장제{chunk['article']}조_part{i+1}"
                
                final_chunks.append({
                    "content": enhanced_content,
                    "chunk_type": chunk["chunk_type"],
                    "chapter": chunk["chapter"],
                    "article": chunk["article"],
                    "chapter_title": chunk["chapter_title"],
                    "article_title": chunk["article_title"],
                    "part_index": i + 1,
                    "total_parts": len(sub_chunks),
                    "chunk_id": chunk_id
                })
        else:
            # 짧은 조는 그대로, 앞에 조 정보만 추가
            if chunk["chunk_type"] == "preamble":
                enhanced_content = f"[서문] {chunk['content']}"
            else:
                enhanced_content = f"[제{chunk['chapter']}장 제{chunk['article']}조 ({chunk['article_title']})] {chunk['content']}"
            
            final_chunks.append({
                "content": enhanced_content,
                "chunk_type": chunk["chunk_type"],
                "chapter": chunk["chapter"],
                "article": chunk["article"],
                "chapter_title": chunk["chapter_title"],
                "article_title": chunk["article_title"],
                "chunk_id": chunk["chunk_id"]
            })
    
    logger.info(f"긴 조 분할 완료: {len(final_chunks)}개 최종 청크")
    return final_chunks

class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = None,
        model_name: str = "text-embedding-3-small",
    ):
        """
        벡터스토어 초기화

        Args:
            persist_directory (str): 벡터스토어를 저장할 디렉토리 경로
            collection_name (str, optional): 사용할 컬렉션 이름 (ETF별 컬렉션 권장)
            model_name (str): OpenAI 임베딩 모델 이름
        """
        # .env 파일 로드
        load_dotenv()

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings(model=model_name)

        # Chroma 클라이언트 설정
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Langchain Chroma 벡터스토어 초기화 (컬렉션이 지정된 경우에만)
        if collection_name:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding,
                client=self.client,
                collection_name=self.collection_name,
            )
        else:
            self.vectorstore = None

    @staticmethod
    def load_pdf_with_contextual_retrieval(
        filepath: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 30,
        add_contextual_summary: bool = True,
        document_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Anthropic의 Contextual Retrieval 방식으로 PDF 파일을 로드하고 청크에 맥락 요약 추가

        Args:
            filepath (str): PDF 파일 경로
            chunk_size (int): 청크 크기
            chunk_overlap (int): 청크 간 중복 크기
            add_contextual_summary (bool): GPT 요약 추가 여부
            document_metadata (dict): 문서 메타데이터 (ETF명, 날짜 등)

        Returns:
            List[Document]: 맥락이 추가된 분할 문서 리스트
        """
        # PDF 로드
        loader = PyPDFLoader(filepath)
        pages = loader.load_and_split()

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        chunks = text_splitter.split_documents(pages)
        
        if not add_contextual_summary:
            return chunks
        
        # 문서 컨텍스트 생성
        document_context = ""
        if document_metadata:
            etf_name = document_metadata.get('etf_name', '')
            ticker = document_metadata.get('ticker', '')
            doc_type = document_metadata.get('doc_type', '')
            date = document_metadata.get('date', '')
            
            document_context = f"이 문서는 {etf_name}({ticker}) ETF의 {doc_type} 문서({date})입니다."
        
        # 각 청크에 맥락 요약 추가
        enhanced_chunks = []
        total_chunks = len(chunks)
        
        logger.info(f"Contextual Retrieval 적용 시작: {total_chunks}개 청크")
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:  # 진행상황 로깅
                logger.info(f"청크 처리 진행: {i+1}/{total_chunks}")
            
            # 맥락 요약 추가
            enhanced_content = add_contextual_summary_to_chunk(
                chunk.page_content, 
                document_context
            )
            
            # 새 Document 생성
            enhanced_chunk = Document(
                page_content=enhanced_content,
                metadata=chunk.metadata
            )
            enhanced_chunks.append(enhanced_chunk)
        
        logger.info(f"Contextual Retrieval 적용 완료: {len(enhanced_chunks)}개 청크")
        return enhanced_chunks

    @staticmethod
    def load_pdf(filepath: str, chunk_size: int = 1000, chunk_overlap: int = 30):
        """
        기존 PDF 로드 방식 (하위 호환성)
        """
        return VectorStore.load_pdf_with_contextual_retrieval(
            filepath=filepath,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_contextual_summary=False
        )

    def add_documents(
        self, documents: List[Document], collection_name: Optional[str] = None
    ):
        """
        문서를 벡터스토어에 추가

        Args:
            documents (List[Document]): 추가할 문서 리스트
            collection_name (str, optional): 컬렉션 이름
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 1) 컬렉션 이름 결정
            target = collection_name or self.collection_name
            if not target:
                raise ValueError("컬렉션 이름이 지정되지 않았습니다.")
            
            logger.info(f"문서 추가 시작: 컬렉션 '{target}', 문서 수: {len(documents)}")

            # 2) 컬렉션 존재 여부 확인 및 생성
            try:
                collection = self.client.get_collection(target)
                logger.info(f"기존 컬렉션 '{target}' 발견 (현재 문서 수: {collection.count()})")
            except (ValueError, Exception) as e:
                logger.info(f"컬렉션 '{target}' 생성 중... (오류: {e})")
                try:
                    # 기존 클라이언트를 사용하여 컬렉션 생성
                    collection = self.client.create_collection(name=target)
                    logger.info(f"새 컬렉션 '{target}' 생성 완료")
                except Exception as create_error:
                    logger.error(f"컬렉션 생성 실패: {create_error}")
                    raise

            # 3) 문서가 있는지 확인
            if not documents:
                logger.warning("추가할 문서가 없습니다.")
                return

            # 4) 해당 컬렉션으로 Chroma 인스턴스 생성 후 문서 추가
            try:
                vs = Chroma(
                    client=self.client,
                    collection_name=target,
                    embedding_function=self.embedding,
                )
                
                # 문서 추가
                vs.add_documents(documents)
                logger.info(f"문서 추가 성공: {len(documents)}개")
                
                # 5) 저장된 문서 수 확인
                col = self.client.get_collection(target)
                total_count = col.count()
                logger.info(f"컬렉션 '{target}' 저장 완료 (총 문서 수: {total_count})")
                print(f"컬렉션 '{target}' 저장 완료 (총 문서 수: {total_count})")
                
            except Exception as add_error:
                logger.error(f"문서 추가 중 오류: {add_error}")
                raise

        except Exception as e:
            logger.error(f"저장 중 치명적 오류 발생: {str(e)}", exc_info=True)
            print(f"저장 중 오류 발생: {str(e)}")
            raise

    def similarity_search(
        self, query: str, k: int = 20, collection_name: Optional[str] = None
    ):
        """
        유사도 검색 수행

        Args:
            query (str): 검색 쿼리
            k (int): 반환할 결과 개수 (기본값 20으로 증가)
            collection_name (str, optional): 검색할 컬렉션 이름

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        if collection_name:
            collection = self.client.get_collection(collection_name)
            vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding,
            )
            return vectorstore.similarity_search(query, k=k)

        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(
        self, search_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Any:  # VectorStoreRetriever 타입 추가
        """
        벡터스토어의 retriever를 반환

        Args:
            search_kwargs (dict, optional): 검색 관련 추가 인자
            **kwargs: 추가 인자

        Returns:
            VectorStoreRetriever: 검색을 위한 retriever 객체
        """
        if search_kwargs is None:
            search_kwargs = {"k": 20}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs, **kwargs)

    def get_etf_collection_name(self, ticker: str) -> str:
        """
        ETF ticker에 대응하는 컬렉션 이름 반환
        
        Args:
            ticker (str): ETF 티커 (예: "069500")
            
        Returns:
            str: 컬렉션 이름 (예: "etf_069500")
        """
        return f"etf_{ticker}"

    def get_etf_retriever(self, ticker: str, search_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        """
        특정 ETF ticker의 retriever를 반환
        
        Args:
            ticker (str): ETF 티커
            search_kwargs (dict, optional): 검색 관련 추가 인자
            **kwargs: 추가 인자
            
        Returns:
            VectorStoreRetriever: 해당 ETF의 retriever 객체
        """
        collection_name = self.get_etf_collection_name(ticker)
        
        try:
            # 컬렉션 존재 확인
            self.client.get_collection(collection_name)
            
            # 해당 컬렉션의 vectorstore 생성
            etf_vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding,
            )
            
            if search_kwargs is None:
                search_kwargs = {"k": 20}
                
            return etf_vectorstore.as_retriever(search_kwargs=search_kwargs, **kwargs)
            
        except ValueError:
            raise ValueError(f"ETF {ticker}의 컬렉션 '{collection_name}'이 존재하지 않습니다.")

    def search_etf_documents(self, ticker: str, query: str, k: int = 20, doc_type: str = None, date: str = None):
        """
        특정 ETF의 문서에서 검색
        
        Args:
            ticker (str): ETF 티커
            query (str): 검색 쿼리
            k (int): 반환할 결과 개수 (기본값 20으로 증가)
            doc_type (str, optional): 문서 타입 필터 (monthly, prospectus, agreement)
            date (str, optional): 날짜 필터
            
        Returns:
            List[Document]: 검색된 문서 리스트
        """
        collection_name = self.get_etf_collection_name(ticker)
        
        try:
            # 컬렉션 존재 확인
            collection = self.client.get_collection(collection_name)
            
            # 해당 컬렉션의 vectorstore 생성
            etf_vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embedding,
            )
            
            # 메타데이터 필터 구성
            filter_dict = {}
            if doc_type:
                filter_dict["doc_type"] = doc_type
            if date:
                filter_dict["date"] = date
            
            # 필터가 있는 경우와 없는 경우 구분
            if filter_dict:
                return etf_vectorstore.similarity_search(
                    query, k=k, filter=filter_dict
                )
            else:
                return etf_vectorstore.similarity_search(query, k=k)
                
        except ValueError:
            raise ValueError(f"ETF {ticker}의 컬렉션 '{collection_name}'이 존재하지 않습니다.")

    def list_etf_collections(self) -> List[str]:
        """
        ETF 관련 컬렉션 목록 반환
        
        Returns:
            List[str]: ETF 컬렉션 이름 목록
        """
        all_collections = self.client.list_collections()
        etf_collections = [col.name for col in all_collections if col.name.startswith("etf_")]
        return etf_collections

    def get_etf_collection_stats(self, ticker: str = None) -> Dict[str, Any]:
        """
        ETF 컬렉션 통계 정보 반환
        
        Args:
            ticker (str, optional): 특정 ETF 티커, None이면 모든 ETF 통계
            
        Returns:
            Dict: 컬렉션 통계 정보
        """
        if ticker:
            # 특정 ETF 통계
            collection_name = self.get_etf_collection_name(ticker)
            try:
                collection = self.client.get_collection(collection_name)
                return {
                    "ticker": ticker,
                    "collection_name": collection_name,
                    "document_count": collection.count(),
                    "metadata": collection.metadata
                }
            except ValueError:
                return {
                    "ticker": ticker,
                    "collection_name": collection_name,
                    "document_count": 0,
                    "error": "컬렉션이 존재하지 않습니다."
                }
        else:
            # 모든 ETF 통계
            etf_collections = self.list_etf_collections()
            stats = {}
            
            for collection_name in etf_collections:
                ticker = collection_name.replace("etf_", "")
                try:
                    collection = self.client.get_collection(collection_name)
                    stats[ticker] = {
                        "collection_name": collection_name,
                        "document_count": collection.count(),
                        "metadata": collection.metadata
                    }
                except Exception as e:
                    stats[ticker] = {
                        "collection_name": collection_name,
                        "document_count": 0,
                        "error": str(e)
                    }
            
            return stats

    @staticmethod
    def load_agreement_pdf_with_enhanced_chunking(
        filepath: str,
        max_tokens: int = 1000,
        add_contextual_summary: bool = True,
        document_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Agreement PDF 파일을 향상된 방식으로 로드
        - 서문 보존
        - 조별 분할 (조 간 overlap 없음)
        - 긴 조 내부 분할 (조 내부에서만 overlap)
        - 조 정보를 청크 앞에 포함
        - Contextual Retrieval 적용

        Args:
            filepath (str): PDF 파일 경로
            max_tokens (int): 최대 토큰 수 (긴 조 분할 기준)
            add_contextual_summary (bool): GPT 요약 추가 여부
            document_metadata (dict): 문서 메타데이터 (ETF명, 날짜 등)

        Returns:
            List[Document]: 향상된 방식으로 분할된 문서 리스트
        """
        logger.info(f"Agreement PDF 향상된 처리 시작: {filepath}")
        
        # 1단계: PDF에서 텍스트 추출
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(filepath)
            pages = loader.load_and_split()
            full_text = "\n".join([page.page_content for page in pages])
            logger.info(f"PDF 텍스트 추출 완료: {len(full_text)}자")
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 실패: {e}")
            raise
        
        # 2단계: 서문 + 조별 분할
        try:
            article_chunks = extract_agreement_with_preamble_and_articles(full_text)
            if not article_chunks:
                logger.warning("Agreement 분할 실패, 기본 청킹 방식 사용")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_tokens,
                    chunk_overlap=100
                )
                basic_chunks = text_splitter.split_documents(pages)
                return basic_chunks
            
            logger.info(f"조별 분할 완료: {len(article_chunks)}개 청크")
        except Exception as e:
            logger.error(f"Agreement 분할 실패: {e}")
            raise
        
        # 3단계: 긴 조 내부 분할
        try:
            final_chunks = split_long_articles(article_chunks, max_tokens)
            logger.info(f"긴 조 분할 완료: {len(final_chunks)}개 최종 청크")
        except Exception as e:
            logger.error(f"긴 조 분할 실패: {e}")
            raise
        
        # 4단계: Document 객체로 변환
        documents = []
        for chunk in final_chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "chunk_type": chunk["chunk_type"],
                    "chapter": chunk["chapter"],
                    "article": chunk["article"],
                    "chapter_title": chunk["chapter_title"],
                    "article_title": chunk["article_title"],
                    "chunk_id": chunk["chunk_id"],
                    "part_index": chunk.get("part_index", 1),
                    "total_parts": chunk.get("total_parts", 1),
                }
            )
            documents.append(doc)
        
        # 5단계: Contextual Retrieval 적용
        if add_contextual_summary and documents:
            try:
                logger.info(f"Contextual Retrieval 적용 시작: {len(documents)}개 청크")
                
                enhanced_documents = []
                for i, doc in enumerate(documents):
                    if i % 5 == 0:
                        logger.info(f"Contextual 처리 진행: {i+1}/{len(documents)}")
                    
                    # 문서별 컨텍스트 생성
                    if document_metadata:
                        etf_name = document_metadata.get('etf_name', '')
                        ticker = document_metadata.get('ticker', '')
                        doc_type = document_metadata.get('doc_type', '')
                        date = document_metadata.get('date', '')
                        base_context = f"이 문서는 {etf_name}({ticker}) ETF의 {doc_type} 문서({date})입니다."
                    else:
                        base_context = "이 문서는 ETF 신탁계약서입니다."
                    
                    # 청크별 세부 컨텍스트
                    if doc.metadata.get("chunk_type") == "preamble":
                        context = f"{base_context} 현재 내용은 신탁계약서의 서문 및 기본 정보입니다."
                    else:
                        chapter = doc.metadata.get("chapter", "")
                        article = doc.metadata.get("article", "")
                        article_title = doc.metadata.get("article_title", "")
                        part_info = ""
                        if doc.metadata.get("total_parts", 1) > 1:
                            part_info = f" (제{article}조의 {doc.metadata.get('part_index', 1)}/{doc.metadata.get('total_parts', 1)} 부분)"
                        
                        context = f"{base_context} 현재 내용은 제{chapter}장 제{article}조({article_title})에 관한 조항{part_info}입니다."
                    
                    # Contextual 요약 추가
                    enhanced_content = add_contextual_summary_to_chunk(
                        doc.page_content,
                        context
                    )
                    
                    # 새 Document 생성 (메타데이터도 업데이트)
                    enhanced_doc = Document(
                        page_content=enhanced_content,
                        metadata={
                            **doc.metadata,
                            # 문서 메타데이터도 추가
                            **(document_metadata if document_metadata else {})
                        }
                    )
                    enhanced_documents.append(enhanced_doc)
                
                logger.info(f"Contextual Retrieval 적용 완료: {len(enhanced_documents)}개 청크")
                return enhanced_documents
                
            except Exception as e:
                logger.error(f"Contextual Retrieval 적용 실패: {e}")
                logger.info("기본 Document 리스트 반환")
                # Contextual 적용 실패 시 기본 문서에 메타데이터만 추가
                for doc in documents:
                    if document_metadata:
                        doc.metadata.update(document_metadata)
                return documents
        
        # Contextual 적용하지 않는 경우
        if document_metadata:
            for doc in documents:
                doc.metadata.update(document_metadata)
        
        logger.info(f"Agreement PDF 처리 완료: {len(documents)}개 청크")
        return documents

def process_pdf_directory(
    vector_store: VectorStore, pdf_dir: str, collection_name: Optional[str] = None
):
    """
    지정된 디렉토리의 새로운 PDF 파일만 처리하여 벡터스토어에 추가

    Args:
        vector_store (VectorStore): 벡터스토어 인스턴스
        pdf_dir (str): PDF 파일들이 있는 디렉토리 경로
        collection_name (str, optional): 저장할 컬렉션 이름
    """
    # 디버깅을 위한 출력 추가
    pdf_path = Path(pdf_dir).joinpath("pdf")
    print(f"PDF 디렉토리 경로: {pdf_path.absolute()}")
    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"발견된 PDF 파일들: {[pdf.name for pdf in pdf_files]}")

    processed_states_path = (
        Path(vector_store.persist_directory) / "processed_states.json"
    )
    print(f"처리 상태 파일 경로: {processed_states_path.absolute()}")

    # 처리된 상태 로드
    processed_states = {}
    if processed_states_path.exists():
        with open(processed_states_path, "r", encoding="utf-8") as f:
            processed_states = json.load(f)

    # 새로운 PDF 파일 찾기
    new_pdf_files = [pdf for pdf in pdf_files if pdf.name not in processed_states]

    if not new_pdf_files:
        print("처리할 새로운 PDF 파일이 없습니다.")
        return

    # 실제 처리할 파일 수를 먼저 출력
    print(f"처리할 새로운 PDF 파일: {len(new_pdf_files)}개")

    for pdf_path in tqdm(new_pdf_files, desc="PDF 처리 중"):
        try:
            documents = VectorStore.load_pdf(str(pdf_path))
            vector_store.add_documents(
                documents=documents, collection_name=collection_name
            )
            # 벡터스토어 처리 상태만 기록
            if pdf_path.name not in processed_states:
                processed_states[pdf_path.name] = {"vectorstore_processed": True}
            else:
                processed_states[pdf_path.name]["vectorstore_processed"] = True

            # 상태 저장
            with open(processed_states_path, "w", encoding="utf-8") as f:
                json.dump(processed_states, f, ensure_ascii=False, indent=2)

            print(f"완료: {pdf_path}")

        except Exception as e:
            print(f"오류 발생 ({pdf_path}): {str(e)}")
