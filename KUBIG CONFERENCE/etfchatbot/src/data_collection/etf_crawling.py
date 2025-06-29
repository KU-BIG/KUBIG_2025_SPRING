"""
삼성자산운용 ETF 라이브러리 크롤러

(1) https://www.samsungfund.com/etf/product/library/pdf.do  접속
(2) '더보기' 버튼을 눌러 전체 ETF 행 노출
(3) 표의 각 <tr> 에서
      · etf_name
      · distribution_link (.xls)
      · agreement_link (.pdf)
      · prospectus_link (.pdf)
      · date  (페이지 상단 날짜)
      · detail_link
(4) 각 detail_link 로 이동해
      · ticker        (종목코드)
      · price_link    (최근 3개월간 기준가 .xls)
      · monthly_link  (월간보고서 .pdf)
(5) MongoDB(etf_library 컬렉션)에
      {etf_name, ticker, date} 복합 유니크 인덱스 → 중복이면 스킵
"""

from __future__ import annotations
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

from selenium.webdriver.support import expected_conditions as EC

from pymongo.errors import DuplicateKeyError

# 프로젝트 내부 의존성
from src.data_collection.db.database_config import get_db_connection
from src.utils.mongodb_utils import MongoDBHandler

# Selenium Driver 초기화
def init_driver(headless: bool = True):
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.113 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )

def setup_unique_index():
    try:
        db = get_db_connection()              # 이미 etf_db 반환하도록 수정됨
        coll = db["etf_library"]              # ETF 전용 컬렉션
        # ticker + date 복합 유니크 인덱스
        coll.create_index(
            [("ticker", 1), ("date", 1)],
            unique=True,
            name="ticker_date_uidx"
        )
        logging.info("Database unique index 설정 완료: etf_library(ticker,date)")
    except Exception as e:
        logging.error(f"인덱스 생성 실패: {e}")
        raise

# 더보기 버튼 반복 → 모든 ETF 행 로드 (개수 제한 없음)

def load_all_rows(driver: webdriver.Chrome,
                  click_timeout: float = 20.0,
                  scroll_retries: int = 10,
                  scroll_pause: float = 0.3):
    ROWS_XPATH = "//table/tbody/tr"
    BTN_XPATH  = "//button[contains(normalize-space(.),'더보기')]"

    # 1) 최소 1개 행 보일 때까지 대기
    WebDriverWait(driver, click_timeout).until(
        EC.presence_of_element_located((By.XPATH, ROWS_XPATH))
    )

    while True:
        rows = driver.find_elements(By.XPATH, ROWS_XPATH)
        current = len(rows)
        logging.info("현재 로우 개수: %d", current)

        # 2) 버튼이 보일 때까지 여러 번 스크롤
        for _ in range(scroll_retries):
            driver.execute_script("window.scrollBy(0, window.innerHeight);")
            time.sleep(scroll_pause)

        try:
            # 3) 스크롤 후 버튼을 찾아 클릭
            btn = WebDriverWait(driver, click_timeout).until(
                EC.element_to_be_clickable((By.XPATH, BTN_XPATH))
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.2)
            btn.click()
            logging.info("더보기 클릭 – 이전 %d행", current)

            # 4) 클릭 후 로우 증가 대기
            WebDriverWait(driver, click_timeout).until(
                lambda d: len(d.find_elements(By.XPATH, ROWS_XPATH)) > current
            )
            time.sleep(scroll_pause)

        except TimeoutException:
            logging.info("더보기 버튼이 더 이상 없음 – 모든 ETF 행 로드 완료: %d행", current)
            break

    final = len(driver.find_elements(By.XPATH, ROWS_XPATH))
    logging.info("최종 로우 개수 확인: %d행", final)

    
# 메인 페이지 → 기본 6개 필드 수집
def parse_library_page(driver: webdriver.Chrome) -> list[dict]:
    # 구체적 XPath로 날짜 입력창 대기
    DATE_XPATH = (
        '//*[@id="root-etf-library"]/div/div/div[2]'
        '/div/div[2]/div/div[2]/div/div/div[1]/div/input'
    )
    date_input = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, DATE_XPATH))
    )
    raw_date = date_input.get_attribute("value").strip()  # e.g. "2025.05.22"
    # "." → "-" 로만 바꾸면 바로 원하는 포맷
    date_str = raw_date.replace(".", "-")               # "2025-05-22"

    ROWS_XPATH = '//*[@id="root-etf-library"]/div/div/div[2]/div/table/tbody/tr'
    rows = driver.find_elements(By.XPATH, ROWS_XPATH)
    results = []

    for row in rows:
        tds = row.find_elements(By.TAG_NAME, "td")
        if len(tds) < 7:
            continue

        etf_name   = tds[0].find_element(By.TAG_NAME, "strong").text.strip()
        detail_link = tds[0].find_element(By.TAG_NAME, "a") \
                             .get_attribute("href")

        def href(idx: int) -> str | None:
            try:
                return tds[idx].find_element(By.TAG_NAME, "a") \
                                .get_attribute("href")
            except:
                return None

        results.append({
            "etf_name":          etf_name,
            "ticker":            "",  # detail 페이지에서 채워짐
            "date":              date_str,
            "detail_link":       detail_link,
            "price_link":        "",  # detail 페이지에서 채워짐
            "distribution_link": href(1),
            "agreement_link":    href(4),
            "prospectus_link":   href(5),
            "monthly_link":      "",  # detail 페이지에서 채워짐
        })

    logging.info("기본 정보 수집 완료: %d건", len(results))
    return results


def enrich_with_detail_info(driver: webdriver.Chrome, record: Dict) -> Dict:
    url = record["detail_link"]
    driver.get(url)

    time.sleep(2.0)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "root-product-view"))
        )
    except TimeoutException:
        logging.warning("디테일 페이지 컨테이너 로드 타임아웃: %s", url)
    time.sleep(1.0)

    ticker = ""
    ticker_xpaths = [
        "//*[@id='root-product-view']/div[2]/div/div/div/div[3]/div/span",
        "//*[@id='root-product-view']/div[3]/div/div/div[2]/section[1]/div/div[2]/div/div/div[1]/div[1]/h3/small",
        "/html/body/div[1]/main/div/div[2]/div/div[2]/div[1]/div[2]/div/span",
        "/html/body/div[1]/main/div/div[2]/div/div/div/div[3]/div/span",
        "//*[@id='root-product-view']//span[contains(@class,'product-ticker')]",
        "//*[@id='root-product-view']//span[contains(@class,'category-stkticker')]",
    ]
    for xp in ticker_xpaths:
        if ticker:
            break
        try:
            el = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.XPATH, xp))
            )
            text = el.text.strip()
            logging.debug("XPath [%s]에서 텍스트 발견: '%s'", xp, text)
            
            # 6자리 숫자 패턴 매칭 (정확히 6자리)
            if re.fullmatch(r"\d{6}", text):
                ticker = text
                logging.info("Ticker 발견 via XPath [%s]: %s", xp, ticker)
                break
            # "(123456)" 형태나 다른 패턴에서 6자리 숫자 추출
            m = re.search(r"\b(\d{6})\b", text)
            if m:
                ticker = m.group(1)
                logging.info("Ticker 발견 via pattern match [%s]: %s", xp, ticker)
                break
        except TimeoutException:
            logging.debug("Ticker XPath 미발견 (%s)", xp)
        except Exception as e:
            logging.debug("Ticker XPath 오류 (%s): %s", xp, e)

    # fallback: 모든 span과 small 태그에서 6자리 숫자 찾기
    if not ticker:
        for tag_name in ["span", "small"]:
            elements = driver.find_elements(By.TAG_NAME, tag_name)
            for el in elements:
                try:
                    text = el.text.strip()
                    if re.fullmatch(r"\d{6}", text):
                        ticker = text
                        logging.info("Ticker 발견 via %s scan: %s", tag_name, ticker)
                        break
                except:
                    continue
            if ticker:
                break

    if not ticker:
        logging.error("Ticker 최종 추출 실패: %s", url)
    record["ticker"] = ticker

    price_link: Optional[str] = None
    price_xpaths = [
        "//*[@id='root-product-view']/div[3]/div/div/div[2]/section[3]/div/div[2]/div/div/a",
        "//*[@id='root-product-view']//a[contains(@href,'/excel_standar.do')]",
        "//*[@id='root-product-view']//a[contains(@href,'excel') and contains(@class,'btn')]",
    ]
    for xp in price_xpaths:
        if price_link:
            break
        try:
            a = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            href = a.get_attribute("href").strip()
            if href and ('/excel_standar.do' in href or 'excel' in href.lower()):
                # 상대 경로인 경우 절대 경로로 변환
                if href.startswith('/'):
                    price_link = f"https://www.samsungfund.com{href}"
                else:
                    price_link = href
                logging.info("Price link 발견 via XPath [%s]: %s", xp, price_link)
                break
        except TimeoutException:
            logging.debug("Price XPath 미발견 (%s)", xp)

    if not price_link:
        logging.warning("기준가 엑셀 링크 최종 미발견: %s", url)
    record["price_link"] = price_link

    monthly_link: Optional[str] = None
    # 맨 아래로 스크롤
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.0)
    try:
        btn_area = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH,
                "//*[@id='root-product-view']//section[2]"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn_area)
        time.sleep(0.5)
    except TimeoutException:
        logging.debug("월간보고서 섹션 스크롤 타임아웃")

    monthly_xpaths = [
        "//*[@id='root-product-view']//a[contains(@href,'/sheet/') and contains(@href,'.pdf')]",
        "//*[@id='root-product-view']//section[2]//a[contains(@href,'.pdf')]",
    ]
    for xp in monthly_xpaths:
        if monthly_link:
            break
        try:
            a = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, xp))
            )
            monthly_link = a.get_attribute("href").strip()
            logging.info("Monthly link 발견 via XPath [%s]: %s", xp, monthly_link)
        except TimeoutException:
            logging.debug("Monthly XPath 미발견 (%s)", xp)

    if not monthly_link:
        logging.warning("월간보고서 링크 최종 미발견: %s", url)
    record["monthly_link"] = monthly_link

    time.sleep(0.5)
    return record


def insert_if_new(handler: MongoDBHandler, data: Dict):
    coll = handler.collection  # handler는 etf_library 컬렉션 바인딩
    try:
        coll.insert_one(data)
        logging.info("INSERT: %s (%s)", data["ticker"], data["date"])
    except DuplicateKeyError:
        logging.info("SKIP(dup): %s (%s)", data["ticker"], data["date"])


def ensure_index(handler: MongoDBHandler):
    handler.collection.create_index(
        [("ticker", 1), ("date", 1)], unique=True, name="ticker_date_uidx"
    )


def crawl_etf_library(headless: bool = True, limit: int = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # DB 연결: etf_db 반환
    db = get_db_connection()
    coll = db["etf_library"]

    # {etf_name, ticker, date} 복합 유니크 인덱스 설정
    coll.create_index(
        [("etf_name", 1), ("ticker", 1), ("date", 1)],
        unique=True,
        name="etf_name_ticker_date_uidx"
    )

    driver = init_driver(headless=headless)
    try:
        driver.get("https://www.samsungfund.com/etf/product/library/pdf.do")
        load_all_rows(driver)

        # 1차 수집: 기본 필드
        records = parse_library_page(driver)
        
        # limit이 설정된 경우 레코드 수 제한
        if limit is not None and limit > 0:
            records = records[:limit]
            logging.info("PRODUCT_DOWNLOAD_LIMIT 적용: %d개 상품으로 제한", limit)

        # 2차 detail 페이지 방문 → ticker, price_link, monthly_link 보강 및 DB 저장
        failed_records = []
        skipped_incomplete = 0
        
        for rec in records:
            try:
                enrich_with_detail_info(driver, rec)
                
                # 필수 필드 검증
                required_fields = ["etf_name", "ticker", "date", "detail_link"]
                missing_fields = [field for field in required_fields if not rec.get(field)]
                
                if missing_fields:
                    logging.warning("필수 필드 누락으로 저장 스킵 - %s: 누락 필드 %s", 
                                  rec.get("etf_name", "Unknown"), missing_fields)
                    skipped_incomplete += 1
                    continue
                
                # 선택적 필드 검증 (경고만)
                optional_fields = ["price_link", "distribution_link", "agreement_link", "prospectus_link", "monthly_link"]
                missing_optional = [field for field in optional_fields if not rec.get(field)]
                if missing_optional:
                    logging.info("선택적 필드 누락 - %s: %s", rec["etf_name"], missing_optional)
                
                try:
                    coll.insert_one(rec)
                    logging.info("INSERT: %s (%s) - %s", rec["etf_name"], rec["ticker"], rec["date"])
                except DuplicateKeyError:
                    logging.info("SKIP(dup): %s (%s) - %s", rec["etf_name"], rec["ticker"], rec["date"])
                    
            except Exception as e:
                logging.error("Detail 처리 실패 – %s : %s", rec["detail_link"], e)
                failed_records.append(rec)

        # 실패한 페이지만 한 번 더 재시도
        if failed_records:
            logging.info("재시도할 실패 레코드 수: %d건", len(failed_records))
            for rec in failed_records[:]:  # 복사본 순회
                try:
                    enrich_with_detail_info(driver, rec)
                    
                    # 재시도에서도 필수 필드 검증
                    required_fields = ["etf_name", "ticker", "date", "detail_link"]
                    missing_fields = [field for field in required_fields if not rec.get(field)]
                    
                    if missing_fields:
                        logging.warning("재시도 - 필수 필드 누락으로 저장 스킵 - %s: 누락 필드 %s", 
                                      rec.get("etf_name", "Unknown"), missing_fields)
                        skipped_incomplete += 1
                        failed_records.remove(rec)
                        continue
                    
                    try:
                        coll.insert_one(rec)
                        logging.info("INSERT(재시도): %s (%s) - %s", rec["etf_name"], rec["ticker"], rec["date"])
                    except DuplicateKeyError:
                        logging.info("SKIP(dup,재시도): %s (%s) - %s", rec["etf_name"], rec["ticker"], rec["date"])
                    failed_records.remove(rec)
                except Exception as e:
                    logging.error("재시도 실패 – %s : %s", rec["detail_link"], e)

        # 최종 결과 요약
        logging.info("=== 크롤링 완료 요약 ===")
        logging.info("필수 필드 누락으로 스킵된 레코드: %d건", skipped_incomplete)
        if failed_records:
            logging.warning("최종 실패 레코드: %d건 남음", len(failed_records))
            for rec in failed_records:
                logging.warning("  - %s (%s)", rec.get("etf_name", "Unknown"), rec["detail_link"])

    finally:
        driver.quit()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser with GUI (for local debugging)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of products to save to MongoDB",
    )
    args = parser.parse_args()

    # 기본: headless=True
    crawl_etf_library(headless=not args.no_headless, limit=args.limit)