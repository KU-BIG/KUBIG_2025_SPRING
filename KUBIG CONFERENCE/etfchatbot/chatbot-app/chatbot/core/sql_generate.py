"""SQL 쿼리 생성 모듈"""
import logging
import sqlite3
from typing import List, Dict
from langchain_openai import ChatOpenAI
from chatbot.prompts.tool_prompts import SQL_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class SQLGenerator:
    """SQL 쿼리 생성기"""
    
    def __init__(self, db_path: str = "data/etf_database.sqlite"):
        self.db_path = db_path
        self.schema_info = self._get_schema_info()
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    
    def _get_schema_info(self) -> str:
        """데이터베이스 스키마 정보 추출"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_parts = []
            for table in tables:
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                schema_parts.append(f"\n테이블: {table}")
                for col in columns:
                    schema_parts.append(f"  - {col[1]} ({col[2]})")
                
                # 샘플 데이터 (최대 2개)
                cursor = conn.execute(f"SELECT * FROM {table} LIMIT 2")
                samples = cursor.fetchall()
                if samples:
                    schema_parts.append(f"  샘플: {samples[0]}")
            
            conn.close()
            return "\n".join(schema_parts)
            
        except Exception as e:
            logger.error(f"스키마 정보 추출 실패: {e}")
            return "스키마 정보를 가져올 수 없습니다."
    
    def generate_query(self, user_question: str, tickers: List[str]) -> str:
        """SQL 쿼리 생성"""
        sql_prompt = SQL_GENERATION_PROMPT.format(
            schema_info=self.schema_info,
            user_question=user_question,
            ticker_list=tickers
        )
        
        try:
            response = self.llm.invoke(sql_prompt)
            query = response.content.strip()
            
            # 쿼리 정리
            if "```sql" in query:
                query = query.split("```sql")[1].split("```")[0].strip()
            elif "```" in query:
                query = query.split("```")[1].strip()
            
            # LIMIT 제거
            if "LIMIT" in query.upper():
                lines = query.split('\n')
                lines = [line for line in lines if 'LIMIT' not in line.upper()]
                query = '\n'.join(lines)
            
            logger.info(f"생성된 SQL: {query}")
            return query
            
        except Exception as e:
            logger.error(f"SQL 생성 실패: {e}")
            return "" 