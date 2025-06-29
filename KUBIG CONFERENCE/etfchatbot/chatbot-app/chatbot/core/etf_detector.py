import logging
import json
import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from chatbot.prompts.tool_prompts import ETF_DETECTION_PROMPT

logger = logging.getLogger(__name__)


class SimpleETFDetector:
    """간단한 ETF 감지기"""
    
    def __init__(self, mapping_path: str = "data/etf_mapping.json"):
        self.mapping_path = mapping_path
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
        self._mapping = None
    
    def load_mapping(self) -> Dict[str, str]:
        """ETF 매핑 로드"""
        if self._mapping is None:
            try:
                with open(self.mapping_path, 'r', encoding='utf-8') as f:
                    self._mapping = json.load(f)
                logger.info(f"ETF 매핑 로드 완료: {len(self._mapping)}개")
            except Exception as e:
                logger.error(f"ETF 매핑 로드 실패: {e}")
                self._mapping = {}
        return self._mapping
    
    def detect_etfs(self, query: str) -> List[str]:
        """쿼리에서 ETF 티커들 감지 - 'all' 또는 특정 티커들 반환"""
        mapping = self.load_mapping()
        
        detection_prompt = ETF_DETECTION_PROMPT.format(
            etf_mapping=json.dumps(mapping, ensure_ascii=False, indent=2),
            user_question=query
        )
        
        try:
            response = self.llm.invoke(detection_prompt)
            ticker_string = response.content.strip()
            
            # "all" 케이스 처리
            if ticker_string.lower() == "all":
                logger.info("ETF 전반적 내용 질문 감지 - 모든 ETF 검색")
                return ["all"]
            
            # 특정 티커들 파싱
            if ticker_string and ticker_string != "":
                tickers = [ticker.strip() for ticker in ticker_string.split(",")]
                tickers = [ticker for ticker in tickers if ticker and ticker.lower() != "all"]  # 빈 문자열 제거
                if tickers:
                    logger.info(f"감지된 특정 ETF 티커: {tickers}")
                    return tickers
            
            logger.info("ETF와 관련 없는 질문 - 일반 답변 모드")
            return []
                
        except Exception as e:
            logger.error(f"ETF 감지 실패: {e}")
            return [] 