# 오늘의집 제품 리뷰 요약봇
## 주요 기능
✅ 오늘의집 제품 페이지 URL 입력시 해당 제품 50개 리뷰 수집
✅ Gemma-7B + 4bit 양자화 기반으로 리뷰의 장점/단점 요약
✅ Mecab 형태소 분석기 기반 키워드 추출 및 관련 리뷰 필터링
✅ Streamlit 기반 챗봇 인터페이스

---
# 실행방법
## 패키지 설치
!pip install selenium
!apt-get update

!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver '/content/drive/MyDrive/Colab Notebooks'
!pip install chromedriver-autoinstaller

!pip install bitsandbytes

!pip install streamlit pyngrok

## Mecab 설치
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab_light_220429.sh

!apt-get update
!apt-get install g++ openjdk-8-jdk
!pip3 install konlpy JPype1-py3
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

## 어플리케이션 실행
streamlit run app.py
애플리케이션 실행 후:
1. 오늘의집 제품 URL 입력
2. 키워드 또는 질문 입력 (선택)
3. 분석 시작 버튼 클릭



