import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional
from langchain_openai import ChatOpenAI
from chatbot.prompts.tool_prompts import DATE_DETECTION_PROMPT
import calendar

logger = logging.getLogger(__name__)


class DateUtils:
    """날짜 관련 유틸리티 - 감지, 검증, 가장 가까운 날짜 찾기"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    
    def detect_dates_from_query(self, query: str) -> Optional[List[str]]:
        """사용자 질문에서 날짜 감지 (LLM 사용)"""
        try:
            # 현재 날짜 정보 추가
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_year_month = now.strftime("%Y-%m") # YYYY-MM 형식 추가
            yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            
            # 지난 일주일 날짜들 생성
            last_week_dates = []
            for i in range(7):
                date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                last_week_dates.append(date)
            last_week_str = ", ".join(reversed(last_week_dates))  # 오래된 날짜부터
            
            # DATE_DETECTION_PROMPT 사용
            formatted_prompt = DATE_DETECTION_PROMPT.format(
                current_date=current_date,
                current_year_month=current_year_month, # 프롬프트에 전달
                yesterday=yesterday,
                week_ago=week_ago,
                last_week_str=last_week_str,
                user_question=query
            )

            response = self.llm.invoke(formatted_prompt)
            result = response.content.strip()
            
            if not result:
                logger.info("LLM이 관련 날짜를 선택하지 않음")
                return None
            
            if result == "all":
                logger.info("LLM이 모든 기간 검색을 결정")
                return ["all"]
            
            # LLM 응답에서 순수한 날짜만 추출 (robust parsing)
            # YYYY-MM-DD 또는 YYYY-MM 형태의 날짜만 추출
            date_pattern = r'\b\d{4}-\d{2}(?:-\d{2})?\b'
            found_dates = re.findall(date_pattern, result)
            
            if not found_dates:
                logger.warning(f"LLM 응답에서 날짜를 추출할 수 없음: {result}")
                return None
            
            # 월 단위 날짜 확장 (YYYY-MM 형식)
            if len(found_dates) == 1 and re.match(r'^\d{4}-\d{2}$', found_dates[0]):
                year_month = found_dates[0]
                logger.info(f"월 단위 날짜 감지됨: {year_month}. 해당 월의 모든 날짜로 확장합니다.")
                year, month = map(int, year_month.split('-'))
                
                # 해당 월의 마지막 날짜 계산
                _, num_days = calendar.monthrange(year, month)
                
                # 월간 보고서 자체(YYYY-MM) + 모든 일자(YYYY-MM-DD)
                expanded_dates = [year_month] 
                for day in range(1, num_days + 1):
                    expanded_dates.append(f"{year_month}-{day:02d}")
                
                logger.info(f"확장된 날짜: {len(expanded_dates)}개")
                return expanded_dates

            logger.info(f"LLM이 결정한 날짜들: {found_dates}")
            return found_dates
                
        except Exception as e:
            logger.error(f"LLM 날짜 선택 실패: {e}")
            return None
    
    def find_closest_date(self, target_date: str, available_dates: List[str]) -> Optional[str]:
        """목표 날짜에 가장 가까운 사용 가능한 날짜 찾기"""
        try:
            # 목표 날짜에서 순수한 날짜만 추출
            date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
            target_match = re.search(date_pattern, target_date)
            
            if target_match:
                clean_target_date = target_match.group(1)
            else:
                # YYYY-MM-DD 형태가 아니면 직접 사용
                if re.match(r'^\d{4}-\d{2}-\d{2}$', target_date):
                    clean_target_date = target_date
                else:
                    logger.warning(f"목표 날짜 형식이 올바르지 않음: {target_date}")
                    return None
            
            target_dt = datetime.strptime(clean_target_date, "%Y-%m-%d")
            
            closest_date = None
            min_diff = float('inf')
            
            for date in available_dates:
                try:
                    # 사용 가능한 날짜도 정제
                    clean_date = date.strip()
                    
                    # YYYY-MM-DD 또는 YYYY-MM 형태 처리
                    if len(clean_date.split('-')) == 2:  # YYYY-MM 형태
                        date_dt = datetime.strptime(clean_date + "-01", "%Y-%m-%d")  # 월의 첫째 날로 변환
                    else:  # YYYY-MM-DD 형태
                        date_dt = datetime.strptime(clean_date, "%Y-%m-%d")
                    
                    diff = abs((target_dt - date_dt).days)
                    if diff < min_diff:
                        min_diff = diff
                        closest_date = clean_date
                        
                except ValueError as e:
                    logger.debug(f"날짜 파싱 실패: {date} - {e}")
                    continue
            
            if closest_date:
                logger.info(f"가장 가까운 날짜 검색 성공: {clean_target_date} → {closest_date} ({min_diff:.0f}일 차이)")
            
            return closest_date
            
        except Exception as e:
            logger.error(f"가장 가까운 날짜 검색 실패: {e}")
            return None
    
    def validate_and_adjust_dates(self, target_dates: List[str], available_dates: List[str]) -> List[str]:
        """목표 날짜들을 검증하고 가장 가까운 날짜로 조정"""
        if not target_dates or "all" in target_dates:
            return target_dates
        
        valid_dates = []
        for date in target_dates:
            if date in available_dates:
                valid_dates.append(date)
            else:
                # 가장 가까운 날짜 찾기
                closest_date = self.find_closest_date(date, available_dates)
                if closest_date:
                    logger.info(f"날짜 조정: {date} → {closest_date}")
                    valid_dates.append(closest_date)
                else:
                    logger.warning(f"날짜 {date}에 대한 대체 날짜를 찾을 수 없음")
        
        # 중복 제거
        return list(set(valid_dates))


# 전역 인스턴스 (싱글톤 패턴)
_date_utils_instance = None

def get_date_utils() -> DateUtils:
    """DateUtils 싱글톤 인스턴스 반환"""
    global _date_utils_instance
    if _date_utils_instance is None:
        _date_utils_instance = DateUtils()
    return _date_utils_instance 