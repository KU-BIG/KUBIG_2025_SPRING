"""Coordinator Agent - OpenAI Functions AgentExecutor를 사용하여 모든 도구 조율"""
import os
import logging
import json
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.tools import BaseTool

# LangSmith 추적 설정
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

from chatbot.tools import (
    ETFDetectorTool,
    DateParserTool,
    SQLGeneratorTool,
    SQLRunnerTool,
    VectorSearchTool
)
from chatbot.prompts.agent_prompts import COORDINATOR_SYSTEM_PROMPT, FINAL_ANSWER_PROMPT

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """ETF 챗봇 코디네이터 에이전트"""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1-mini",
                 temperature: float = 0,
                 verbose: bool = True,
                 enable_langsmith: bool = True):
        """
        Args:
            model_name: OpenAI 모델 이름
            temperature: 생성 온도 (0 = 결정적)
            verbose: 상세 로그 출력 여부
            enable_langsmith: LangSmith 추적 활성화 여부
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.enable_langsmith = enable_langsmith
        
        # LangSmith 설정
        self.langsmith_client = None
        self.tracer = None
        if self.enable_langsmith:
            self._setup_langsmith()
        
        # 도구들 초기화
        self.tools = self._initialize_tools()
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=False  # CoT를 위해 스트리밍 비활성화
        )
        
        # 메모리 초기화
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 에이전트 생성
        self.agent_executor = self._create_agent_executor()
    
    def _setup_langsmith(self):
        """LangSmith 추적 설정"""
        try:
            # 환경 변수 확인
            api_key = os.getenv("LANGCHAIN_API_KEY")
            project_name = os.getenv("LANGCHAIN_PROJECT", "etf-chatbot-agent")
            
            if api_key:
                self.langsmith_client = Client(api_key=api_key)
                self.tracer = LangChainTracer(project_name=project_name)
                logger.info(f"LangSmith 추적 활성화됨 - 프로젝트: {project_name}")
            else:
                logger.warning("LANGCHAIN_API_KEY 환경변수가 설정되지 않음. LangSmith 추적이 비활성화됩니다.")
                self.enable_langsmith = False
        except Exception as e:
            logger.error(f"LangSmith 설정 실패: {e}")
            self.enable_langsmith = False
    
    def _initialize_tools(self) -> List[BaseTool]:
        """모든 도구 초기화"""
        tools = [
            ETFDetectorTool(),
            DateParserTool(),
            SQLGeneratorTool(),
            SQLRunnerTool(),
            VectorSearchTool()
        ]
        logger.info(f"초기화된 도구: {[tool.name for tool in tools]}")
        return tools
    
    def _create_agent_executor(self) -> AgentExecutor:
        """OpenAI Functions Agent 생성"""
        # 프롬프트 템플릿 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", COORDINATOR_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # OpenAI Functions Agent 생성
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # 콜백 설정
        callbacks = []
        if self.enable_langsmith and self.tracer:
            callbacks.append(self.tracer)
        
        # AgentExecutor 생성
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            handle_parsing_errors=True,
            return_intermediate_steps=True,  # CoT 추적을 위해
            max_iterations=10,  # 무한 루프 방지
            max_execution_time=120,  # 2분 타임아웃
            callbacks=callbacks  # LangSmith 콜백 추가
        )
        
        logger.info("OpenAI Functions Agent 생성 완료")
        return agent_executor
    
    def _format_tool_outputs(self, intermediate_steps) -> str:
        """도구 실행 결과를 포맷팅하여 최종 답변 생성에 사용"""
        tool_outputs = []
        
        for action, observation in intermediate_steps:
            tool_name = action.tool
            tool_input = action.tool_input
            
            # 입력이 딕셔너리인 경우 문자열로 변환
            if isinstance(tool_input, dict):
                tool_input_str = json.dumps(tool_input, ensure_ascii=False)
            else:
                tool_input_str = str(tool_input)
            
            # 결과가 너무 길면 잘라내기
            if len(str(observation)) > 1000:
                observation_str = str(observation)[:1000] + "... (중략)"
            else:
                observation_str = str(observation)
            
            output = f"### {tool_name}\n"
            output += f"**입력**: {tool_input_str}\n"
            output += f"**결과**: {observation_str}\n"
            
            tool_outputs.append(output)
        
        return "\n".join(tool_outputs)
    
    def run(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """질문 처리 및 답변 생성
        
        Args:
            question: 사용자 질문
            session_id: LangSmith 추적을 위한 세션 ID
            
        Returns:
            dict: {
                "output": 최종 답변,
                "intermediate_steps": 중간 단계들 (CoT 추적),
                "tool_outputs": 도구 실행 결과들,
                "trace_url": LangSmith 추적 URL (사용 가능한 경우)
            }
        """
        try:
            logger.info(f"질문 처리 시작: {question}")
            
            # LangSmith 메타데이터 설정
            metadata = {
                "question": question,
                "model": self.model_name,
                "temperature": self.temperature
            }
            if session_id:
                metadata["session_id"] = session_id
            
            # 콜백 설정 (LangSmith 추적 포함)
            callbacks = []
            if self.enable_langsmith and self.tracer:
                callbacks.append(self.tracer)
            
            # Agent 실행
            result = self.agent_executor.invoke(
                {"input": question},
                config={
                    "callbacks": callbacks,
                    "metadata": metadata,
                    "tags": ["etf_chatbot", "coordinator_agent"]
                }
            )
            
            # 중간 단계 로깅 (CoT 추적)
            if "intermediate_steps" in result:
                logger.info("=== Chain-of-Thought 추적 ===")
                for i, (action, observation) in enumerate(result["intermediate_steps"]):
                    logger.info(f"단계 {i+1}: {action.tool} - {action.tool_input}")
                    logger.info(f"결과: {str(observation)[:200]}...")
            
            # 도구 출력 정리
            tool_outputs = {}
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    tool_outputs[action.tool] = observation
            
            # 최종 답변 생성 (필요시 FINAL_ANSWER_PROMPT 사용)
            # 이미 agent가 최종 답변을 생성했으므로 그대로 사용
            final_output = result.get("output", "답변을 생성할 수 없습니다.")
            
            # LangSmith 추적 URL 생성 (가능한 경우)
            trace_url = None
            if self.enable_langsmith and hasattr(result, 'run_id'):
                trace_url = f"https://smith.langchain.com/o/{self.langsmith_client.project_name}/traces/{result.run_id}"
            
            response = {
                "output": final_output,
                "intermediate_steps": result.get("intermediate_steps", []),
                "tool_outputs": tool_outputs
            }
            
            if trace_url:
                response["trace_url"] = trace_url
            
            return response
            
        except Exception as e:
            logger.error(f"Agent 실행 실패: {e}")
            return {
                "output": f"처리 중 오류가 발생했습니다: {str(e)}",
                "intermediate_steps": [],
                "tool_outputs": {}
            }
    
    def generate_final_answer(self, question: str, intermediate_steps) -> str:
        """도구 실행 결과를 바탕으로 최종 답변 생성"""
        try:
            # 도구 실행 결과 포맷팅
            tool_outputs_str = self._format_tool_outputs(intermediate_steps)
            
            # 최종 답변 생성 프롬프트 포맷팅
            formatted_prompt = FINAL_ANSWER_PROMPT.format(
                tool_outputs=tool_outputs_str,
                question=question
            )
            
            # LLM으로 최종 답변 생성
            response = self.llm.invoke(formatted_prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"최종 답변 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def clear_memory(self):
        """대화 메모리 초기화"""
        self.memory.clear()
        logger.info("대화 메모리 초기화됨") 