import os
import subprocess
import logging
from typing import Any
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# 새로운 Agent 아키텍처 import
from chatbot.agent import CoordinatorAgent

# S3 유틸리티 import
from chatbot.core.s3_utils import load_vectorstore_from_s3, save_to_s3

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

def setup_aws_credentials_if_available():
    """Jenkins 환경에서 AWS credentials 설정 시도, 서빙이 아닌 오직 로컬 디버깅용입니다."""
    try:
        # Jenkins에서 설정된 AWS credentials 확인
        if os.path.exists('/root/.aws/credentials'):
            logger.info("기존 AWS credentials 파일 발견")
            return True
            
        # 환경변수로 AWS credentials 설정 시도
        result = subprocess.run(['aws', 'configure', 'list'], 
                               capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("AWS CLI 설정 확인됨")
            return True
        else:
            logger.warning(f"AWS CLI 설정 확인 실패: {result.stderr}")
            return False
            
    except Exception as e:
        logger.warning(f"AWS credentials 확인 중 오류: {e}")
        return False

# ChatBot 초기화 시 AWS credentials 확인
setup_aws_credentials_if_available()

def create_agent_chain():
    """Agent 기반 RAG 체인 생성"""
    logger.info("=== Agent 기반 체인 생성 시작 ===")
    
    # CoordinatorAgent 초기화
    agent = CoordinatorAgent(
        model_name="gpt-4.1-mini",
        temperature=0,
        verbose=True
    )
    
    # Agent를 LangChain 체인처럼 사용할 수 있도록 래핑
    def agent_wrapper(question: str) -> str:
        """Agent를 실행하고 최종 답변만 반환"""
        try:
            result = agent.run(question)
            
            # 중간 단계 로깅 (디버깅용)
            if result.get("intermediate_steps"):
                logger.info(f"Agent가 사용한 도구 수: {len(result['intermediate_steps'])}")
                for i, (action, _) in enumerate(result["intermediate_steps"]):
                    logger.info(f"  {i+1}. {action.tool}")
            
            return result["output"]
            
        except Exception as e:
            logger.error(f"Agent 실행 실패: {e}")
            return f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
    
    # LangChain 체인 형태로 반환
    chain = RunnableLambda(agent_wrapper)
    
    logger.info("Agent 기반 체인 생성 완료")
    return chain

def chatbot():
    """챗봇 초기화 및 체인 반환 - Agent 아키텍처 사용"""
    try:
        # 1. S3에서 VectorDB 다운로드 및 초기화
        logger.info("=== S3에서 VectorDB 초기화 시작 ===")
        client = load_vectorstore_from_s3()
        logger.info("S3 VectorDB 초기화 완료")
        
        # 2. Agent 기반 체인 생성
        qa_chain = create_agent_chain()
        
        logger.info("챗봇 초기화 완료 - Agent + Tool 아키텍처 (CoT 포함)")
        logger.info("사용 가능한 도구: ETF감지, 날짜감지, SQL생성, SQL실행, 벡터검색")
        return qa_chain
        
    except Exception as e:
        logger.error(f"챗봇 초기화 실패: {str(e)}")
        raise

# 공통 export
__all__ = ["chatbot", "create_agent_chain", "save_to_s3"] 