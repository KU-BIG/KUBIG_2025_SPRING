"""ETF 챗봇 프롬프트 패키지

챗봇 시스템에서 사용하는 다양한 프롬프트 템플릿들을 제공합니다.
"""

from .tool_prompts import (
    SIMPLE_ETF_PROMPT,
    ETF_DETECTION_PROMPT,
    DATE_DETECTION_PROMPT,
    SQL_GENERATION_PROMPT
)

from .agent_prompts import (
    COORDINATOR_SYSTEM_PROMPT,
    FINAL_ANSWER_PROMPT
)

__all__ = [
    # ETF 기본 프롬프트
    "SIMPLE_ETF_PROMPT",
    "ETF_DETECTION_PROMPT",
    "DATE_DETECTION_PROMPT",
    "SQL_GENERATION_PROMPT",
    
    # 에이전트 프롬프트
    "COORDINATOR_SYSTEM_PROMPT",
    "FINAL_ANSWER_PROMPT"
] 