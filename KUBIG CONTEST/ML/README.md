# 🚗 Project: 전기차 가격 예측 해커톤: 데이터로 EV를 읽다! (데이콘)
https://dacon.io/competitions/official/236424/overview/description
## ⚡ 팀명: Watt's the Price?
20기 이정제, 21기 김수환, 21기 윤채영 

---

## 📊 변수 설명  
- ID: 차량별 고유 ID
- 제조사: 차량 제조사(H사, B사, K사, A사, T사, P사, V사 총 7개의 제조사 중 하나)
- 모델: 차량 모델명(총 21개의 모델 중 하나)
- 차량상태: Brand New(신차), Nearly New(준신차), Pre-Owned(중고차)
- 배터리용량: 잔존 배터리용량(kWh 단위)
- 구동방식: AWD(사륜구동), FWD(전륜구동), RWD(후륜구동)
- 주행거리(km): 누적 주행거리
- 보증기간(년): 무상 수리 및 서비스 기간
- 사고이력: No, Yes
- 연식(년): 0, 1, 2로 이루어짐(ex. 연식이 1: 출시된 지 1년 된 차량)
- 가격(백만원): 전기차 가격
---

## 🔍 EDA (탐색적 데이터 분석)  
- 📈 **데이터 분포 및 상관관계 분석**  
- 🎨 **변수 시각화를 통한 인사이트 도출**  
- 🛠 **파생변수 생성**  
  - 모델 평균 가격  
  - 연식 대비 주행거리(모델2에서는 미사용) 
- 📏 **표준화**  

---

## 🏗 모델링 과정  

### 🛠 모델1  
- 🔄 **전처리**: 라벨 인코딩, Ridge 모델 기반 결측치 대체값 생성 등 
- 🔍 **하이퍼파라미터 탐색**: `RandomSearchCV` 활용  
- 🚘 **모델링**: `RandomForestRegressor` 사용  

---

### 🛠 모델2 (모델1의 한계 개선)  
- ❌ 특정 차량 모델(**4번**, 14번, 15번)에서 **예측력 저하** 문제 발견  
- 💡 **해결책**: 데이터셋을 `'Model 4'`와 `'그 외 모델'`로 분리하여 **개별 모델** 적용  

#### 🔄 전처리  
- 라벨 인코딩  
- Ridge 모델 기반 결측치 대체값 생성  
- 이상치 제거  

#### 🚘 모델링  
- **'Model 4'** → `Adaboost` 사용  
- **'그 외 모델'** → `RandomForestRegressor` 사용  

---

## 📈 결과  
| 모델 | Public Score | Private Score |
|------|-------------|--------------|
| 모델1 | 1.0170744974 | 1.2206657563 |
| 모델2 | **0.9213119795** | **1.2055857718** (개선됨) |

✅ **모델2가 성능이 더 우수함을 확인!**  
