# Team2 🏠중 탐구

## subject 
[데이콘](https://dacon.io/competitions/official/236439/data) 부동산 허위매물 분류 해커톤: 가짜를 색출하라!

## members
19기 심서현, 20기 김채원, 21기 김지엽, 21기 엄희문

## Goal
집을 직접 방문하거나 사진을 보지 않고 수치적인 요소만으로 허위매물 예측하기

## Data
- train.csv: 허위매물여부 포함 16개의 feature로 구성, 2452개
- test.csv: 허위매물여부 없이 15개의 feature로 구성, 612개

## Processing
> ### EDA, Feature Engineering
> - 주어진 데이터의 특성들을 분석
> - 등록일이 오래될수록 허위매물 확률이 높기 때문에 이를 반영하기 위해서 '게재일수' = ‘최신 일자' - ‘등록 일자' 라는 수치형 변수를 포함하여 분석을 진행
> - 특정 변수들에 결측치가 존재하면 허위매물일 확률이 높기 때문에 구분하여 라벨 인코딩
> - '중개사무소'라는 중요한 변수를 타겟 인코딩하기에는 거래량의 차이가 있는 중개사무소들을 구분하지 못하기에 'laplace smoothing'과 '신뢰등급'을 사용하여 적용

> ### Adapting model
> ##### Tabular 데이터를 처리하는 다양한 머신러닝 툴들을 적용
> - Random Forest (Optuna)
> - CatBoost (SMOTE + Optuna)
> - LGBM (SMOTE + roc-auc 기반 Bayesian Optimization)
> - XGBoost (Optuna)
> ##### DL 세션에서 배운 딥러닝의 모델의 개념과 구조도 적용
> - MLP (KNNImputer, Ordinal Encoder 적용, Optuna + Smote + Adnam)
> - TabNet (Optuna + SMOTE)
> - TabPFN (SMOTE)

## Results
- marco F1 score로 성능 계산
