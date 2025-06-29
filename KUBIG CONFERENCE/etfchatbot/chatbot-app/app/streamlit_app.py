import streamlit as st
import requests
import uuid
import time

# 페이지 설정
st.set_page_config(
    page_title="ETF 챗봇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API 설정
API_BASE_URL = "http://localhost:8000"

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 제목
st.title("3000Agent")
st.markdown("---")

# 사이드바에 세션 정보
with st.sidebar:
    st.header("세션 정보")
    st.text(f"세션 ID: {st.session_state.session_id[:8]}...")
    
    if st.button("새 세션 시작"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### 사용 가능한 질문 예시:
    - "069500 ETF의 구성종목을 알려줘"
    - "최근 일주일간 ETF 성과를 비교해줘"
    - "ETF 투자 전략을 설명해줘"
    """)

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("ETF에 대해 궁금한 것을 물어보세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 봇 응답 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 로딩 표시
        with st.spinner("Agent가 열심히 분석 중입니다..."):
            try:
                # API 호출
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/chat/sendMessage",
                    json={
                        "request": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=120  # 2분 타임아웃
                )
                
                if response.status_code == 200:
                    result = response.json()
                    bot_response = result["response"]
                    
                    # LangSmith 추적 URL이 있으면 표시
                    if "trace_url" in result:
                        st.success(f"[추적 보기]({result['trace_url']})")
                    
                else:
                    bot_response = f"오류가 발생했습니다. (상태 코드: {response.status_code})"
                    
            except requests.exceptions.Timeout:
                bot_response = "응답 시간이 초과되었습니다. 다시 시도해주세요."
            except requests.exceptions.ConnectionError:
                bot_response = "서버에 연결할 수 없습니다. FastAPI 서버가 실행 중인지 확인해주세요."
            except Exception as e:
                bot_response = f"예상치 못한 오류가 발생했습니다: {str(e)}"
        
        # 응답 표시
        message_placeholder.markdown(bot_response)
    
    # 봇 응답을 히스토리에 추가
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# 하단 정보
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("대화 수", len([m for m in st.session_state.messages if m["role"] == "user"]))

with col2:
    # API 서버 상태 확인
    try:
        health_response = requests.get(f"{API_BASE_URL}/ping", timeout=5)
        if health_response.status_code == 200:
            st.metric("서버 상태", "온라인")
        else:
            st.metric("서버 상태", "오프라인")
    except:
        st.metric("서버 상태", "연결 불가")

with col3:
    st.metric("세션", f"#{st.session_state.session_id[:8]}") 