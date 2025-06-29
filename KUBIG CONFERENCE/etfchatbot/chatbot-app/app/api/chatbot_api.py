import sys
import os
import logging
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
import asyncio

from chatbot import chatbot

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 로거 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ETF Chatbot API with LangSmith Tracing")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델 정의
class SendMessageRequest(BaseModel):
    request: str
    session_id: Optional[str] = Field(default=None, description="세션 ID (추적용)")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_url: Optional[str] = Field(default=None, description="LangSmith 추적 URL")

# QA Chain 초기화
qa_chain = chatbot()

# ThreadPoolExecutor 초기화
executor = ThreadPoolExecutor(max_workers=10)

async def invoke_chatbot(request: str, session_id: str = None):
    """챗봇 호출 (LangSmith 추적 포함)"""
    loop = asyncio.get_event_loop()
    
    # Agent에서 직접 호출하여 추적 정보 얻기
    from chatbot.agent import CoordinatorAgent
    
    # 기존 qa_chain 대신 agent 직접 사용
    agent = CoordinatorAgent(enable_langsmith=True)
    
    def run_agent():
        return agent.run(request, session_id=session_id)
    
    return await loop.run_in_executor(executor, run_agent)

@app.post("/api/v1/chat/sendMessage", response_model=ChatResponse)
async def send_message(request: SendMessageRequest):
    try:
        # 세션 ID 생성 (제공되지 않은 경우)
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(f"Chat request received - Session: {session_id}, Question: {request.request[:100]}...")
        
        # 챗봇 호출
        result = await invoke_chatbot(request.request, session_id)
        
        response_data = {
            "response": result["output"],
            "session_id": session_id
        }
        
        # LangSmith 추적 URL 포함 (있는 경우)
        if "trace_url" in result:
            response_data["trace_url"] = result["trace_url"]
            logger.info(f"LangSmith trace available: {result['trace_url']}")
        
        logger.info(f"Response generated - Session: {session_id}")
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Chat processing error - Session: {session_id}, Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    return {"status": "running", "langsmith_enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true"}

@app.get("/health")
async def health_check():
    """상세 헬스 체크 (LangSmith 연결 상태 포함)"""
    health_status = {
        "status": "healthy",
        "api": "running",
        "langsmith": {
            "enabled": os.getenv("LANGCHAIN_TRACING_V2") == "true",
            "api_key_set": bool(os.getenv("LANGCHAIN_API_KEY")),
            "project": os.getenv("LANGCHAIN_PROJECT", "etf-chatbot-agent")
        }
    }
    
    return health_status

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.chatbot_api:app", host="0.0.0.0", port=8000, reload=True)
