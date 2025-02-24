멀티모달을 이용한 알츠하이머 예측
=========================
# 1. Motivation
* MRI 영상을 이용한, 단일 modality 기반의 진단 모델은 제한된 정보만을 사용하기 때문에, 알츠하이머와 같은 복합적인 질환의 진단 및 예측에 한계를 보일 것입니다.
* 그래서 text 형태의 환자 데이터를 추가하여 풍부한 임상 정보를 통합적으로 활용하고, 분류 정확도를 높이는 것을 목표로 하였습니다.
* 비슷한 MRI 사진이어도, text 형태의 환자 데이터에 따라 알츠하이머 예측을 다르게 할 것이라 가정했습니다.
# 2. Dataset
## OASIS-1 Dataset
1. 데이터 규모 및 피험자 연령대
* 총 416명의, 18세부터 96세까지의 피험자를 대상으로 한 T1 weighted MRI 영상을 포함
* 모두 오른손잡이이며, 남성과 여성을 모두 포함하고 있음
* 정상 인지 기능부터, 경도(mild)에서 중등도(moderate)의 알츠하이머 질환을 가진 피험자를 포함
2. 개발 목적
* 치매 조기 진단 알고리즘 개발
* 분류(classification), 혹은 회귀(regression) 모델의 학습용
* segmentation 모델의 학습에도 많이 사용됨
3. Data
* 1.5T Siemens Vision MRI 스캐너에서 T1 weighted MPRAGE 시퀀스를 이용해 촬영
![022421371941352](https://github.com/user-attachments/assets/2333ca60-cd07-4774-a0f3-420364dc1f7e)

* 촬영된 사람의 성별, 나이, 치매 정도 등 다양한 정보를 알 수 있는 보고서 포함
<img width="868" alt="Screenshot 2025-02-24 at 9 46 21 PM" src="https://github.com/user-attachments/assets/be01087a-16d8-46ed-93db-3391da363109" />

https://sites.wustl.edu/oasisbrains/

    Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults. Marcus, DS, Wang, TH, Parker, J, Csernansky, JG, Morris, JC, Buckner, RL. Journal of Cognitive Neuroscience, 19, 1498-1507. doi: 10.1162/jocn.2007.19.9.1498
# 3. Method
## 1. Text data

## 2. MRI data

## 3. Multimodal embedding

# 4. Results


