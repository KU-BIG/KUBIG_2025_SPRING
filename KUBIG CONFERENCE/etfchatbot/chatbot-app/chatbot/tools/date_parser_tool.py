"""Date Parser Tool - DateUtils를 래핑하는 LangChain Tool"""
import logging
from typing import Type, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from chatbot.core.date_utils import get_date_utils, DateUtils

logger = logging.getLogger(__name__)


class DateParserArgs(BaseModel):
    """날짜 감지 도구의 입력 스키마"""
    question: str = Field(description="사용자의 질문")
    etf_tickers: str = Field(
        description="관련 ETF 티커들 (콤마로 구분)", 
        default=""
    )


class DateParserTool(BaseTool):
    """사용자 질문에서 날짜 정보 추출 도구"""
    
    name: str = "detect_dates_from_query"
    description: str = """사용자 질문에서 날짜 정보를 추출합니다.
    - "오늘", "어제" 같은 상대적 표현을 절대 날짜로 변환
    - "최근 일주일", "지난주" 같은 기간을 날짜 범위로 변환
    - 구체적 날짜는 그대로 사용
    - 모든 기간인 경우: "all" 반환
    - 날짜 관련 없으면: "none" 반환
    반환값은 콤마로 구분된 날짜 리스트 (YYYY-MM-DD 형식)입니다."""
    
    args_schema: Type[BaseModel] = DateParserArgs
    date_utils: Optional[DateUtils] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.date_utils = get_date_utils()
    
    def _run(
        self, 
        question: str, 
        etf_tickers: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """날짜 감지 실행"""
        try:
            if not self.date_utils:
                self.date_utils = get_date_utils()
            
            # LangSmith 메타데이터 추가
            if run_manager:
                run_manager.on_text(
                    f"날짜 감지 시작: {question[:50]}...",
                    verbose=True
                )
                
            # ETF 티커 파싱
            tickers = []
            if etf_tickers and etf_tickers != "none":
                tickers = [t.strip() for t in etf_tickers.split(",") if t.strip()]
            
            # 날짜 감지
            detected_dates = self.date_utils.detect_dates_from_query(question)
            
            if not detected_dates:
                logger.info("날짜 관련 정보 없음")
                result = "none"
            else:
                result = ",".join(detected_dates)
                logger.info(f"감지된 날짜: {result}")
            
            # LangSmith에 결과 기록
            if run_manager:
                run_manager.on_text(
                    f"날짜 감지 완료: {result}",
                    verbose=True
                )
                # 메타데이터 추가
                run_manager.on_tool_end(
                    result,
                    metadata={
                        "detected_dates": detected_dates or [],
                        "question_length": len(question),
                        "tickers_provided": len(tickers),
                        "detection_type": "specific" if result != "none" and result != "all" else result
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"error: {str(e)}"
            logger.error(f"날짜 감지 실패: {e}")
            
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