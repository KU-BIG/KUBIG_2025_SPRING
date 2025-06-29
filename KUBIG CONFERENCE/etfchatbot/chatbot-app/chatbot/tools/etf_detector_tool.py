"""ETF Detector Tool - SimpleETFDetector를 래핑하는 LangChain Tool"""
import logging
from typing import List, Type, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from chatbot.core.etf_detector import SimpleETFDetector

logger = logging.getLogger(__name__)


class ETFDetectorArgs(BaseModel):
    """ETF 감지 도구의 입력 스키마"""
    question: str = Field(description="사용자의 질문")


class ETFDetectorTool(BaseTool):
    """ETF 티커 감지 도구"""
    
    name: str = "detect_etfs"
    description: str = """사용자 질문에서 ETF 티커를 감지합니다.
    - 특정 ETF가 언급된 경우: 해당 티커 반환 (예: ["069500", "226490"])
    - ETF 전반적인 내용인 경우: ["all"] 반환
    - ETF와 관련 없는 경우: [] 반환
    반환값은 콤마로 구분된 티커 리스트입니다."""
    
    args_schema: Type[BaseModel] = ETFDetectorArgs
    detector: Optional[SimpleETFDetector] = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = SimpleETFDetector()
    
    def _run(
        self, 
        question: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """ETF 감지 실행"""
        try:
            if not self.detector:
                self.detector = SimpleETFDetector()
            
            # LangSmith 메타데이터 추가
            if run_manager:
                run_manager.on_text(
                    f"ETF 감지 시작: {question[:50]}...",
                    verbose=True
                )
            
            detected_etfs = self.detector.detect_etfs(question)
            
            if not detected_etfs:
                logger.info("ETF와 관련 없는 질문 감지")
                result = "none"
            else:
                result = ",".join(detected_etfs)
                logger.info(f"감지된 ETF: {result}")
            
            # LangSmith에 결과 기록
            if run_manager:
                run_manager.on_text(
                    f"ETF 감지 완료: {result}",
                    verbose=True
                )
                # 메타데이터 추가
                run_manager.on_tool_end(
                    result,
                    metadata={
                        "detected_etfs": detected_etfs,
                        "question_length": len(question),
                        "detection_type": "specific" if result != "none" and result != "all" else result
                    }
                )
            
            return result
            
        except Exception as e:
            error_msg = f"error: {str(e)}"
            logger.error(f"ETF 감지 실패: {e}")
            
            # LangSmith에 에러 기록
            if run_manager:
                run_manager.on_tool_error(e)
            
            return error_msg
    
    async def _arun(self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """비동기 실행 (현재는 동기 실행으로 fallback)"""
        return self._run(question, run_manager) 