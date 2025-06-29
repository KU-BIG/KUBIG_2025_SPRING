"""Vector Search Tool - ETFVectorRetriever.search를 래핑하는 LangChain Tool"""
import logging
from typing import Type, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from chatbot.core.vector_retriever import ETFVectorRetriever

logger = logging.getLogger(__name__)


class VectorSearchArgs(BaseModel):
    """벡터 검색 도구의 입력 스키마"""
    query: str = Field(description="검색 질의")
    etf_tickers: str = Field(
        description="대상 ETF 티커들 (콤마로 구분, 'all'이면 모든 ETF)",
        default="all"
    )
    dates: str = Field(
        description="검색할 날짜들 (콤마로 구분, 'all'이면 전체 기간)",
        default="all"
    )
    k: int = Field(
        description="반환할 문서 수",
        default=20
    )


class VectorSearchTool(BaseTool):
    """PDF 문서에서 관련 정보를 검색하는 도구 (BM25 + 벡터 하이브리드)"""
    
    name: str = "search"
    description: str = """ChromaDB에서 ETF 관련 PDF 문서를 검색합니다.
    - BM25(키워드) + 벡터(의미) 하이브리드 검색 사용
    - ETF별 컬렉션에서 검색 (etf_069500, etf_226490 등)
    - 날짜 필터링 지원
    - 투자설명서, 월간보고서, 신탁계약서 등의 문서 검색
    반환값: 검색된 문서 내용과 메타데이터를 포함한 텍스트"""
    
    args_schema: Type[BaseModel] = VectorSearchArgs
    retriever: Optional[ETFVectorRetriever] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retriever = ETFVectorRetriever()
    
    def _run(
        self, 
        query: str, 
        etf_tickers: str = "all", 
        dates: str = "all", 
        k: int = 20,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """벡터 검색 실행"""
        try:
            if not self.retriever:
                self.retriever = ETFVectorRetriever()
            
            # LangSmith 메타데이터 추가
            if run_manager:
                run_manager.on_text(
                    f"벡터 검색 시작 - 쿼리: {query[:50]}..., 티커: {etf_tickers}, 날짜: {dates}",
                    verbose=True
                )
                
            # 티커 파싱
            tickers = []
            if etf_tickers and etf_tickers != "none":
                if etf_tickers == "all":
                    tickers = ["all"]
                else:
                    tickers = [t.strip() for t in etf_tickers.split(",") if t.strip()]
            
            # 날짜 파싱
            date_list = None
            if dates and dates != "none":
                if dates == "all":
                    date_list = ["all"]
                else:
                    date_list = [d.strip() for d in dates.split(",") if d.strip()]
            
            # 검색 실행
            docs = self.retriever.search(
                query=query,
                tickers=tickers,
                dates=date_list,
                k=k
            )
            
            if not docs:
                logger.info("검색 결과 없음")
                result = "no_documents_found"
                
                if run_manager:
                    run_manager.on_text(
                        "벡터 검색 결과 없음",
                        verbose=True
                    )
                    run_manager.on_tool_end(
                        result,
                        metadata={
                            "query_length": len(query),
                            "tickers": tickers,
                            "dates": date_list,
                            "k": k,
                            "documents_found": 0
                        }
                    )
                
                return result
            
            # 결과 포맷팅
            results = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                source = metadata.get("source", "unknown")
                page = metadata.get("page", "unknown")
                date = metadata.get("date", "unknown")
                doc_type = metadata.get("type", "unknown")
                
                result_text = f"""
[문서 {i}]
출처: {source}
페이지: {page}
날짜: {date}
유형: {doc_type}
내용: {doc.page_content[:500]}...
"""
                results.append(result_text)
            
            final_result = "\n".join(results)
            logger.info(f"벡터 검색 완료: {len(docs)}개 문서")
            
            # LangSmith에 결과 기록
            if run_manager:
                run_manager.on_text(
                    f"벡터 검색 완료: {len(docs)}개 문서 발견",
                    verbose=True
                )
                # 메타데이터 추가
                run_manager.on_tool_end(
                    final_result,
                    metadata={
                        "query_length": len(query),
                        "tickers": tickers,
                        "dates": date_list,
                        "k": k,
                        "documents_found": len(docs),
                        "average_content_length": sum(len(doc.page_content) for doc in docs) / len(docs) if docs else 0,
                        "search_type": "hybrid_bm25_vector"
                    }
                )
            
            return final_result
            
        except Exception as e:
            error_msg = f"error: {str(e)}"
            logger.error(f"벡터 검색 실패: {e}")
            
            # LangSmith에 에러 기록
            if run_manager:
                run_manager.on_tool_error(e)
            
            return error_msg
    
    async def _arun(
        self, 
        query: str, 
        etf_tickers: str = "all", 
        dates: str = "all", 
        k: int = 20,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """비동기 실행 (현재는 동기 실행으로 fallback)"""
        return self._run(query, etf_tickers, dates, k, run_manager) 