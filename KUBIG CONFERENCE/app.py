import streamlit as st
import time
import re
import pandas as pd
import torch
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import traceback
from konlpy.tag import Mecab


# --- 크롤러 클래스 ---
class TodayHouseSimpleCrawler:
    def __init__(self, url):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('lang=ko_KR')
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 5)
        self.url = url
        self.seen_reviews = set()

    def collect_reviews(self, max_reviews=50):
        self.driver.get(self.url)

        try:
            review_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[href="#production-selling-review"]'))
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", review_button)
            self.driver.execute_script("arguments[0].click();", review_button)
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article.production-review-item"))) # p.css-1nu11iv
        except Exception as e:
            print(f"[ERROR] 리뷰 요소 로딩 실패: {e}")
            self.driver.quit()
            return []

        collected_reviews = []

        while len(collected_reviews) < max_reviews:
            try:
                self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "article.production-review-item"))) # p.css-1nu11iv
                tags = self.driver.find_elements(By.CSS_SELECTOR, "p.production-review-item__description") # p.css-1nu11iv

                for tag in tags:
                    if len(collected_reviews) >= max_reviews:
                        break
                    try:
                        review_text = tag.text.strip()
                        if review_text and review_text not in self.seen_reviews:
                            collected_reviews.append(review_text)
                            self.seen_reviews.add(review_text)
                    except Exception as e:
                        print(f"[WARN] 리뷰 파싱 실패: {e}")
                        continue
                try:
                    current_page = int(self.driver.find_element(By.CSS_SELECTOR, "button.R16_p").text.strip())
                    next_btn = self.driver.find_element(By.XPATH,f"//button[@class='_3b4ci' and normalize-space(text())='{current_page + 1}']")

                    self.driver.execute_script("arguments[0].click();", next_btn)
                    self.wait.until(EC.staleness_of(tags[0]))
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article.production-review-item")))

                except Exception as e:
                    print(f"[INFO] 다음 페이지가 없거나 클릭 실패: {e}")
                    break

            except Exception as e:
                print(f"[ERROR] 리뷰 수집 중 오류 발생:")
                traceback.print_exc()
                break

        self.driver.quit()
        return collected_reviews

# --- 리뷰 크롤링 함수 ---
def crawl_todayhouse_50_reviews(url):
    match = re.search(r"/productions/(\d+)", url)
    if not match:
        raise ValueError("올바른 오늘의집 제품 URL이 아닙니다.")

    crawler = TodayHouseSimpleCrawler(url)
    reviews = crawler.collect_reviews(max_reviews=50)

    df = pd.DataFrame([r.replace("\n", " ").strip() for r in reviews], columns=["Review"])

    return df

# --- LoRA 및 4bit 양자화 설정 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

@st.cache_resource
def load_lora_model():
    model_name = "google/gemma-7b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # 자동으로 GPU/CPU 혼합할당
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.half()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        return_full_text=False
    )

    return pipe


pipe = load_lora_model()

# --- 요약 관련 함수 ---
def build_prompt(reviews, sentiment, max_prompt_length=8000):
    joined = ""
    limited_reviews = []
    for r in reviews:
        if len(joined) + len(r) + 1 > max_prompt_length:
            break
        limited_reviews.append(r)
        joined += r + "\n"

    reviews_text = "\n".join(limited_reviews)
    if sentiment == "both":
        prompt = f"""
          당신은 제품 리뷰를 분석하는 전문가입니다.
          다음은 한 제품에 대한 실제 사용자 리뷰 모음입니다.

          리뷰를 분석하여 다음 지침에 따라 핵심 내용을 요약하세요:

          1. **비슷한 의견은 하나로 묶고**, 중복 없이 정리해주세요.
          2. **긍정적인 점**과 **아쉬운 점**으로 나누어 요약하세요.
          3. 각 항목은 **간결한 한 문장**으로 작성해주세요.
          4. 제품과 무관한 내용이나 모호한 표현은 제외해주세요.

          [출력 예시 형식]

          - 긍정적인 점:

          - 아쉬운 점:

          [리뷰 목록]
          f"{reviews_text}\n\n"

          [요약 결과]

      """
        return prompt

def generate_summary(prompt, max_new_tokens=300, temperature=0.7):
    result = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    return result[0]["generated_text"].strip()

def split_reviews(reviews, chunk_size=50):
    return [reviews[i:i+chunk_size] for i in range(0, len(reviews), chunk_size)]

def summarize_chunks(chunks, sentiment):
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            prompt = build_prompt(chunk, sentiment)
            summary = generate_summary(prompt)
            summaries.append(summary)
            torch.cuda.empty_cache()
        except Exception as e:
            st.error(f"[ERROR] Part {i+1} - {sentiment} 요약 실패: {e}")
    return " ".join(summaries)

# --- 키워드 추출 함수 ---
mecab = Mecab()
stop_words = {"이거", "저거", "그거", "제품", "좀", "약간", "조금", "되나요", "해주세요", "같아요", "어때요"}

def extract_keywords_from_query_mecab(query):
    pos = mecab.pos(query)
    keywords = []
    for word, tag in pos:
        if tag.startswith("N") or tag == "XR":
            if word not in stop_words:
                keywords.append(word)
        elif tag in ["VV", "VA"]:
            if word.endswith("다"):
                word = word[:-1]
            if word not in stop_words and len(word) > 1:
                keywords.append(word)
    return list(set(keywords))

# --- 리뷰 필터링 함수 ---
def filter_reviews_by_keywords(df, keywords):
    pattern = "|".join(map(re.escape, keywords))
    filtered_df = df[df["Review"].str.contains(pattern, case=False, na=False)]
    return filtered_df


# --- Streamlit UI ---
import streamlit as st

st.set_page_config(page_title="오늘의집 리뷰 요약봇", layout="wide")

st.markdown("<h1 style='color:#004466;'>오늘의집 리뷰 요약봇</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;'>제품 리뷰를 수집하고 요약해 드립니다. 특정 키워드와 관련된 리뷰가 궁금하시다면 키워드를 함께 입력 후 분석 버튼을 클릭해주세요. </p>", unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    st.markdown("<h3 style='color:#00557f;'>1. 입력</h3>", unsafe_allow_html=True)
    url = st.text_input("제품 URL")
    query = st.text_input("키워드 또는 질문")
    start = st.button("분석 시작", type="primary")

with right:
    if start and url:
        with st.spinner("리뷰 수집 중..."):
            df = crawl_todayhouse_50_reviews(url)  # 기존 함수 사용
        st.success("리뷰 수집 완료")

        st.markdown("<h3 style='color:#1c3f60;'>2. 요약 결과</h3>", unsafe_allow_html=True)
        reviews = df["Review"].dropna().tolist()
        chunks = split_reviews(reviews, 50)
        summary = summarize_chunks(chunks, "both")
        st.text_area("요약 결과", summary, height=300)

        if query:
            st.markdown("<h3 style='color:#754c24;'>3. 키워드 분석</h3>", unsafe_allow_html=True)
            keywords = extract_keywords_from_query_mecab(query)
            filtered_df = filter_reviews_by_keywords(df, keywords)
            match = len(filtered_df)
            total = len(df)
            ratio = (match / total) * 100

            st.markdown(f"<p style='color:#aa0000;'>• 키워드: <strong>{', '.join(keywords)}</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#555555;'>• {total}개 중 <strong>{match}개</strong> 포함 ({ratio:.1f}%)</p>", unsafe_allow_html=True)
            st.dataframe(filtered_df)

        with st.expander("전체 리뷰 보기"):
            st.dataframe(df)
