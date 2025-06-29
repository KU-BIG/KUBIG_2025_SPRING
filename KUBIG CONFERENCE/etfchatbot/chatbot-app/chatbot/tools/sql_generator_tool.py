"""SQL Generator Tool - SQLGenerator.generate_query를 래핑하는 LangChain Tool"""
import logging
from typing import Type, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from chatbot.core.sql_generate import SQLGenerator

logger = logging.getLogger(__name__)


class SQLGeneratorArgs(BaseModel):
    """SQL 생성 도구의 입력 스키마"""
    question: str = Field(description="사용자의 질문")
    etf_tickers: str = Field(
        description="관련 ETF 티커들 (콤마로 구분)",
        default=""
    )


class SQLGeneratorTool(BaseTool):
    """SQL 쿼리 생성 도구"""
    
    name: str = "generate_query"
    description: str = """ETF 데이터베이스 조회를 위한 SQL 쿼리를 생성합니다.
    사용 가능한 테이블:
    - etf_prices: 가격, NAV, 수익률, 거래량, 프리미엄/디스카운트 정보
    - etf_distributions: 구성종목, 포트폴리오 비중 정보
    반환값: SQL SELECT 쿼리 문자열"""
    
    args_schema: Type[BaseModel] = SQLGeneratorArgs
    sql_generator: Optional[SQLGenerator] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sql_generator = SQLGenerator()
    
    def _run(
        self, 
        question: str, 
        etf_tickers: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """SQL 쿼리 생성 실행"""
        try:
            if not self.sql_generator:
                self.sql_generator = SQLGenerator()
            
            # LangSmith 메타데이터 추가
            if run_manager:
                run_manager.on_text(
                    f"SQL 쿼리 생성 시작: {question[:50]}...",
                    verbose=True
                )
                
            # ETF 티커 파싱
            tickers = []
            if etf_tickers and etf_tickers not in ["none", "all"]:
                tickers = [t.strip() for t in etf_tickers.split(",") if t.strip()]
            elif etf_tickers == "all":
                # 모든 ETF를 위한 처리 (실제로는 WHERE 절 없이 처리)
                tickers = []
            
            # SQL 쿼리 생성
            sql_query = self.sql_generator.generate_query(question, tickers)
            
            if not sql_query:
                logger.warning("SQL 쿼리 생성 실패")
                error_msg = "error: 쿼리 생성 실패"
                
                if run_manager:
                    run_manager.on_tool_error(Exception("쿼리 생성 실패"))
                
                return error_msg
            
            logger.info(f"생성된 SQL 쿼리: {sql_query[:100]}...")
            
            # LangSmith에 결과 기록
            if run_manager:
                run_manager.on_text(
                    f"SQL 쿼리 생성 완료: {sql_query[:100]}...",
                    verbose=True
                )
                # 메타데이터 추가
                run_manager.on_tool_end(
                    sql_query,
                    metadata={
                        "query_length": len(sql_query),
                        "question_length": len(question),
                        "tickers_count": len(tickers),
                        "tickers": tickers,
                        "query_type": "SELECT" if "SELECT" in sql_query.upper() else "unknown"
                    }
                )
            
            return sql_query
            
        except Exception as e:
            error_msg = f"error: {str(e)}"
            logger.error(f"SQL 쿼리 생성 실패: {e}")
            
            # LangSmith에 에러 기록
            if run_manager:
                run_manager.on_tool_error(e)
            
            return error_msg
    
    async def _arun(
        self, 
        question: str, 
        etf_tickers: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """비동기 실행 (현재는 동기 실행으로 fallback)"""
        return self._run(question, etf_tickers, run_manager) 