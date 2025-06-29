"""SQL Runner Tool - SQLRunner.execute_query를 래핑하는 LangChain Tool"""
import logging
import json
from typing import Type, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from chatbot.core.sql_run import SQLRunner

logger = logging.getLogger(__name__)


class SQLRunnerArgs(BaseModel):
    """SQL 실행 도구의 입력 스키마"""
    sql_query: str = Field(description="실행할 SQL 쿼리")


class SQLRunnerTool(BaseTool):
    """SQL 쿼리 실행 도구"""
    
    name: str = "run_query"
    description: str = """SQL 쿼리를 실행하고 결과를 반환합니다.
    - SELECT 쿼리만 실행 가능
    - 결과는 JSON 형식으로 반환
    - 데이터가 없으면 가장 가까운 날짜로 자동 재검색
    - Distribution 데이터는 토큰 절약을 위해 자동 필터링됨"""
    
    args_schema: Type[BaseModel] = SQLRunnerArgs
    sql_runner: Optional[SQLRunner] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sql_runner = SQLRunner()
    
    def _run(
        self, 
        sql_query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """SQL 쿼리 실행"""
        try:
            if not self.sql_runner:
                self.sql_runner = SQLRunner()
            
            # LangSmith 메타데이터 추가
            if run_manager:
                run_manager.on_text(
                    f"SQL 쿼리 실행 시작: {sql_query[:100]}...",
                    verbose=True
                )
                
            # SQL 쿼리 실행
            results = self.sql_runner.execute_query(sql_query)
            
            if not results:
                logger.info("SQL 쿼리 결과 없음")
                result = "no_data"
                
                if run_manager:
                    run_manager.on_text(
                        "SQL 쿼리 결과 없음",
                        verbose=True
                    )
                    run_manager.on_tool_end(
                        result,
                        metadata={
                            "query_length": len(sql_query),
                            "result_count": 0,
                            "status": "no_data"
                        }
                    )
                
                return result
            
            # 결과를 JSON 문자열로 변환
            # 토큰 절약을 위해 최대 50개 레코드만 반환
            original_count = len(results)
            if len(results) > 50:
                logger.info(f"결과 수 제한: {len(results)}개 → 50개")
                results = results[:50]
            
            result_json = json.dumps(results, ensure_ascii=False, indent=2)
            logger.info(f"SQL 실행 완료: {len(results)}개 레코드")
            
            # LangSmith에 결과 기록
            if run_manager:
                run_manager.on_text(
                    f"SQL 실행 완료: {len(results)}개 레코드 반환",
                    verbose=True
                )
                # 메타데이터 추가
                run_manager.on_tool_end(
                    result_json,
                    metadata={
                        "query_length": len(sql_query),
                        "original_result_count": original_count,
                        "returned_result_count": len(results),
                        "was_truncated": original_count > 50,
                        "status": "success"
                    }
                )
            
            return result_json
            
        except Exception as e:
            error_msg = f"error: {str(e)}"
            logger.error(f"SQL 쿼리 실행 실패: {e}")
            
            # LangSmith에 에러 기록
            if run_manager:
                run_manager.on_tool_error(e)
            
            return error_msg
    
    async def _arun(
        self, 
        sql_query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """비동기 실행 (현재는 동기 실행으로 fallback)"""
        return self._run(sql_query, run_manager) 