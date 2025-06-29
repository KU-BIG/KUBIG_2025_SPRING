"""ETF 챗봇 패키지

Agent 기반 ETF 챗봇 시스템:
- Agent 아키텍처: CoordinatorAgent가 모든 도구 조율
- Chain-of-Thought 추론: 복잡한 질문 처리 개선
- Tool 기반 모듈화: 각 기능을 독립적인 Tool로 래핑
"""

# 메인 진입점
from .chatbot import chatbot, create_agent_chain, save_to_s3

# 핵심 기능 모듈들
from .core import (
    # ETF 감지
    SimpleETFDetector,
    
    # 날짜 처리
    DateUtils, 
    get_date_utils,
    
    # SQL 관련
    SQLGenerator, 
    SQLRunner,
    
    # 벡터 검색
    ETFVectorRetriever,
    load_embedding_model,
    
    # S3 연동
    load_vectorstore_from_s3,
    setup_chromadb_client,
    save_to_s3
)

# 에이전트 시스템
from .agent import CoordinatorAgent

# 프롬프트 템플릿
from .prompts import (
    SIMPLE_ETF_PROMPT,
    ETF_DETECTION_PROMPT,
    DATE_DETECTION_PROMPT,
    SQL_GENERATION_PROMPT,
    COORDINATOR_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT
)

__all__ = [
    # 메인 진입점
    "chatbot",
    "create_agent_chain",
    
    # 핵심 기능 모듈
    "SimpleETFDetector",
    "DateUtils", 
    "get_date_utils",
    "SQLGenerator", 
    "SQLRunner",
    "ETFVectorRetriever",
    "load_embedding_model",
    
    # S3 연동
    "load_vectorstore_from_s3",
    "setup_chromadb_client",
    "save_to_s3",
    
    # 에이전트 시스템
    "CoordinatorAgent",
    
    # 프롬프트 템플릿
    "SIMPLE_ETF_PROMPT",
    "ETF_DETECTION_PROMPT",
    "DATE_DETECTION_PROMPT", 
    "SQL_GENERATION_PROMPT",
    "COORDINATOR_SYSTEM_PROMPT",
    "FINAL_ANSWER_PROMPT"
] 