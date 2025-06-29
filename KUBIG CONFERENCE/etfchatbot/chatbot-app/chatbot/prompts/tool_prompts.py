"""도구 관련 프롬프트 템플릿

ETF 챗봇 시스템에서 사용하는 다양한 도구 관련 프롬프트 템플릿들을 정의합니다.
"""

# 최종 답변 생성을 위한 메인 프롬프트
SIMPLE_ETF_PROMPT = """당신은 ETF 투자 전문가입니다.

제공된 정보를 바탕으로 질문에 정확하고 유용한 답변을 제공하세요:
- 정형 데이터: 데이터베이스의 가격, 구성종목, 수익률 등의 수치 정보
- 비정형 데이터: PDF 문서의 투자설명서, 월간보고서, 신탁계약서 내용

답변 시 다음을 포함하세요:
- 질문에 대한 간략한 요약
- 관련 표나 차트가 있다면 포함
- 상세한 분석과 설명
- 정보 출처 명시

한국어로 자연스럽고 이해하기 쉽게 답변해주세요.

**컨텍스트:**
{context}

**질문:** {question}

**답변:**"""


# ETF 감지를 위한 프롬프트
ETF_DETECTION_PROMPT = """다음 질문에서 언급된 ETF의 티커 코드를 찾아서 리스트로 반환하세요.

사용 가능한 ETF 매핑:
{etf_mapping}

질문: "{user_question}"

판단 규칙:
1. 특정 ETF가 명시된 경우: 해당 티커 코드 반환 (예: 069500, 226490)
2. ETF 전반적 내용, 모든 ETF, ETF 비교 등의 질문: "all" 반환
3. ETF와 관련 없는 질문: 빈 문자열 반환

예시:
- "069500 ETF의 구성종목을 알려줘" → 069500
- "모든 ETF의 성과를 비교해줘" → all
- "ETF란 무엇인가요?" → all
- "ETF 투자 전략을 알려줘" → all
- "오늘 날씨는 어때요?" → (빈 문자열)

응답 형식:
- 특정 티커: 069500, 226490 (콤마로 구분)
- 모든 ETF: all
- 관련 없음: (빈 문자열)

반드시 위 형식으로만 응답하세요."""


# 날짜 감지를 위한 프롬프트
DATE_DETECTION_PROMPT = """현재 날짜: {current_date}

사용자 질문에서 필요한 날짜를 선택하세요.

사용자 질문: "{user_question}"

규칙:
1. "오늘", "현재", "지금" → {current_date}
2. "어제", "yesterday" → {yesterday}  
3. 상대적 기간 표현을 절대 날짜로 변환 (현재 날짜 기준)
4. 월말/월초 경계를 올바르게 처리
5. 연속된 날짜 범위를 모두 포함
6. 구체적 날짜는 그대로 사용
7. 일반적 질문이면 "all"
8. 관련 날짜가 없으면 빈 문자열 ""
9. **월 전체를 의미하는 경우 (예: "6월", "이번 달"): 반드시 `YYYY-MM` 형식 하나만 반환하세요.**

예시:
질문: "오늘자 투자종목구성을 알려줘"
답변: {current_date}

질문: "어제 ETF 성과는?"
답변: {yesterday}

질문: "이번 달 ETF 시황을 요약해줘"
답변: {current_year_month}

질문: "5월달의 월간 보고서서를 요약해줘"
답변: 2025-05

질문: "2025-05-03일 데이터를 보여줘"
답변: 2025-05-03

질문: "최근 일주일간 ETF 성과를 보여줘"
답변: {last_week_str}

질문: "지난달부터 이번달까지의 데이터"
답변: {current_year_month}-01,..., {current_year_month}

질문: "지난주부터 어제까지의 데이터"
답변: {week_ago}, ..., {yesterday}

질문: "ETF의 일반적인 특징을 설명해줘"
답변: all

반드시 다음 형식으로만 응답하세요:
- 특정 달 하나: 2025-05
- 특정 날짜 하나: 2025-05-30
- 여러 달: 2025-04, 2025-05
- 여러 날짜: 2025-04-30, 2025-05-30 
- 모든 기간: all
- 관련 없음: 

설명은 하지 말고 날짜만 반환하세요."""


# SQL 쿼리 생성을 위한 프롬프트
SQL_GENERATION_PROMPT = """ETF 데이터베이스에서 정보를 조회하는 SQL 쿼리를 생성하세요.

데이터베이스 스키마:
{schema_info}

질문: {user_question}
관련 ETF 티커: {ticker_list}

## 중요한 가이드라인:

### 1. **테이블 선택 가이드**

**etf_prices 테이블 사용**:
- 가격, NAV, 수익률 관련 질문
- 거래량, 프리미엄/디스카운트 질문
- 시장 성과, 추적 오차 관련 질문

**etf_distributions 테이블 사용**:
- 구성종목, 포트폴리오 비중 관련 질문
- 상위 보유 종목, 섹터 분석 질문
- 종목별 투자 비중 비교 질문

**JOIN 사용**:
- 가격과 구성종목을 함께 분석하는 복합 질문
- ETF 성과와 주요 보유 종목의 관계 분석

### 2. **쿼리 패턴 예시**

**A. 가격/성과 관련 (etf_prices):**
```sql
SELECT ticker, etf_name, date, nav, market_price, premium_discount, volume
FROM etf_prices 
WHERE ticker IN ([ticker_list]) AND date = '[specific_date]'
ORDER BY ticker
```

**B. 구성종목 비중 비교 (etf_distributions):**
```sql
SELECT ticker, etf_name, stock_code, stock_name, weight_percent
FROM etf_distributions 
WHERE ticker IN ([ticker_list]) AND date = '[specific_date]'
ORDER BY ticker, weight_percent DESC
```

**C. 각 ETF별 상위 N개 종목 (etf_distributions):**
```sql
SELECT * FROM (
    SELECT ticker, etf_name, stock_code, stock_name, weight_percent,
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY weight_percent DESC) as rn
    FROM etf_distributions
    WHERE ticker IN ([ticker_list]) AND date = '[specific_date]'
) ranked
WHERE rn <= [N개수]
ORDER BY ticker, weight_percent DESC
```

**D. 복합 분석 (JOIN):**
```sql
SELECT p.ticker, p.etf_name, p.nav, p.market_price, 
       d.stock_name, d.weight_percent
FROM etf_prices p
JOIN etf_distributions d ON p.ticker = d.ticker AND p.date = d.date
WHERE p.ticker IN ([ticker_list]) AND p.date = '[specific_date]'
  AND d.weight_percent >= [threshold_percent]
ORDER BY p.ticker, d.weight_percent DESC
```

### 3. **금지 사항**
- 구성종목 질문에서 `GROUP BY ticker, etf_name` 사용 (전체 합계가 아닌 개별 종목 필요)
- LIMIT 사용 금지
- 테이블을 혼동하여 잘못된 컬럼 사용

### 4. **기타 규칙**
- 여러 ETF가 있으면 모두 포함
- 날짜 조건은 WHERE 절에 포함
- 적절한 ORDER BY로 결과 정렬
- **실제 쿼리에서는 [placeholder]를 구체적인 값으로 대체**

질문 유형을 정확히 파악하여 적절한 테이블을 선택하고 SELECT 쿼리만 생성하세요:"""


# 모듈 내보내기
__all__ = [
    "SIMPLE_ETF_PROMPT",
    "ETF_DETECTION_PROMPT",
    "DATE_DETECTION_PROMPT",
    "SQL_GENERATION_PROMPT"
] 