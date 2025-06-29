import warnings
import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier # LogisticRegression 대신 SGDClassifier 임포트
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , classification_report # classification_report 추가
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from typing import Dict, Tuple, List, Optional
import utils4 as utils # 유틸리티 함수 임포트
import sys # 명령줄 인수에 접근하기 위한 sys 모듈 임포트

# scikit-learn 경고를 억제하여 깔끔한 출력을 만듭니다.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        # 클라이언트 데이터 로드 (이진 레이블)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val = utils.load_client_data(client_id)
        print(f"클라이언트 {self.client_id}: 로드된 X_train에는 {self.X_train.shape[1]}개의 특성이 있습니다.")

        # 기본 모델을 정의합니다. (XGBoost 및 CatBoost)
        self.xgb_model = XGBClassifier(
            n_estimators=455,
            max_depth=6,
            learning_rate=0.08653415292886991,
            subsample=0.8425998022058797,
            colsample_bytree=0.8996549142330391,
            random_state=42,
        )

        self.cat_model = CatBoostClassifier(
            iterations=520,
            depth=6,
            learning_rate=0.09980327654097068,
            l2_leaf_reg=4.545010679457599,
            random_seed=42,
            verbose=0,
        )

        # 연합될 최종 추정기(메타 학습자)를 SGDClassifier로 정의합니다. (변경됨: LogisticRegression -> SGDClassifier)
        # 로지스틱 회귀를 위해 `loss='log_loss'`를 사용합니다.
        # `warm_start=True`는 `partial_fit`을 사용할 때 이전에 훈련된 모델의 상태를 유지하게 합니다.
        self.meta_learner = SGDClassifier(loss='log_loss', random_state=42, warm_start=True, eta0=0.01, learning_rate='constant')

        # 메타 학습자의 특성 수는 각 기본 모델의 predict_proba 출력에 따라 결정됩니다.
        # 이진 분류의 경우, 각 기본 모델은 2개의 확률(클래스 0, 클래스 1)을 출력합니다.
        # 따라서 2개의 기본 모델이 있으므로 2 * 2 = 4개의 메타 특성이 됩니다.
        num_meta_features = 2 * 2 
        
        print(f"클라이언트 {self.client_id}: {num_meta_features}개의 특성으로 메타 학습자 초기화 중.")

        # meta_learner의 coef_, intercept_ 및 classes_를 직접 초기화합니다.
        # `partial_fit`의 첫 호출에서 `classes_`가 올바르게 설정되어야 하므로 중요합니다.
        self.meta_learner.coef_ = np.zeros((1, num_meta_features))
        self.meta_learner.intercept_ = np.zeros(1)
        self.meta_learner.classes_ = np.array([0, 1])

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        메타 학습자(SGDClassifier)의 매개변수를 반환합니다.
        메타 학습자의 매개변수만 연합합니다.
        """
        print(f"클라이언트 {self.client_id}: 메타 학습자 매개변수를 {self.meta_learner.coef_.shape} 형태로 보냅니다.")
        return [self.meta_learner.coef_, self.meta_learner.intercept_]

    def set_parameters(self, parameters: List[np.ndarray]):
        """
        메타 학습자(SGDClassifier)의 매개변수를 설정합니다.
        """
        # 서버로부터 받은 전역 매개변수로 클라이언트의 메타 학습자를 업데이트합니다.
        self.meta_learner.coef_ = parameters[0]
        self.meta_learner.intercept_ = parameters[1]
        
        # SGDClassifier의 `classes_` 속성이 일관되게 유지되도록 합니다.
        if not hasattr(self.meta_learner, 'classes_') or not np.array_equal(self.meta_learner.classes_, np.array([0, 1])):
            self.meta_learner.classes_ = np.array([0, 1])
        
        # `n_features_in_`가 설정되었는지 확인합니다 (partial_fit에 의해 설정됨)
        if not hasattr(self.meta_learner, 'n_features_in_') or self.meta_learner.n_features_in_ == 0:
            self.meta_learner.n_features_in_ = self.meta_learner.coef_.shape[1]


    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        로컬에서 메타 학습자를 훈련합니다.
        (변경됨: StackingClassifier 전체 훈련 -> 기본 모델 훈련 후 메타 학습자 partial_fit)
        """
        self.set_parameters(parameters) # 전역 매개변수로 메타 학습자를 업데이트합니다.
        print(f"클라이언트 {self.client_id}: 로컬 훈련 시작 (기본 모델 및 메타 학습자 partial_fit)...")
        
        # 1. 로컬 훈련 데이터에서 기본 모델을 훈련합니다.
        # 이 기본 모델은 각 라운드마다 로컬 데이터에서 훈련됩니다.
        self.xgb_model.fit(self.X_train, self.y_train)
        self.cat_model.fit(self.X_train, self.y_train)
        print(f"클라이언트 {self.client_id}: 기본 모델 훈련 완료.")

        # 2. 훈련된 기본 모델의 predict_proba를 사용하여 메타 특성을 생성합니다.
        # 각 기본 모델은 이진 분류를 위해 (샘플 수, 2) 확률을 출력합니다.
        meta_features_xgb = self.xgb_model.predict_proba(self.X_train)
        meta_features_cat = self.cat_model.predict_proba(self.X_train)
        
        # 메타 학습자를 위한 메타 특성을 연결합니다.
        X_meta_train = np.hstack([meta_features_xgb, meta_features_cat])
        
        print(f"클라이언트 {self.client_id}: 메타 학습자 훈련을 위한 메타 특성 형태: {X_meta_train.shape}")

        # 3. 생성된 메타 특성으로 메타 학습자(SGDClassifier)를 부분적으로 훈련합니다.
        # Server에서 local_epochs를 설정하지 않은 경우 기본값 2로 설정합니다.
        epochs_per_round = int(config.get("local_epochs", 2)) # 기본값 2 에포크 설정
        print(f"클라이언트 {self.client_id}: 라운드 당 {epochs_per_round} 에포크로 메타 학습자 partial_fit 훈련 중.")

        for i in range(epochs_per_round):
            # partial_fit을 원하는 에포크 수만큼 반복 호출합니다.
            # classes 매개변수는 partial_fit의 첫 호출에서만 필수이지만,
            # 안전하게 매번 포함하여 모델이 항상 올바른 클래스를 인식하도록 합니다.
            self.meta_learner.partial_fit(X_meta_train, self.y_train, classes=np.array([0, 1]))
            #print(f"클라이언트 {self.client_id}: partial_fit 에포크 {i+1}/{epochs_per_round} 완료.")

        print(f"클라이언트 {self.client_id}: 메타 학습자 partial_fit {epochs_per_round}번 훈련 완료.")
        
        print(f"클라이언트 {self.client_id}: 로컬 훈련 종료.")
        return self.get_parameters({}), len(self.X_train), {}
    

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        검증 데이터에서 로컬로 메타 학습자를 평가합니다.
        """
        self.set_parameters(parameters) # 전역 매개변수로 메타 학습자를 업데이트합니다.
        
        # 로컬에서 훈련된 기본 모델을 사용하여 평가 데이터에 대한 메타 특성을 생성합니다.
        meta_features_xgb_val = self.xgb_model.predict_proba(self.X_val)
        meta_features_cat_val = self.cat_model.predict_proba(self.X_val)
        X_meta_val = np.hstack([meta_features_xgb_val, meta_features_cat_val])

        # 메타 학습자로 예측을 수행합니다.
        y_pred = self.meta_learner.predict(X_meta_val)

        # 평가 지표 계산 및 각 클래스별 지표 포함 
        accuracy = accuracy_score(self.y_val, y_pred)
        
        # classification_report를 사용하여 각 클래스별 정밀도, 재현율, F1-점수 추출
        report = classification_report(self.y_val, y_pred, output_dict=True, zero_division=0)

        precision_0 = report['0']['precision']
        recall_0 = report['0']['recall']
        f1_0 = report['0']['f1-score']

        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_1 = report['1']['f1-score']

        print(f"클라이언트 {self.client_id}: 평가 결과 - 정확도: {accuracy:.4f}")
        print(f"클라이언트 {self.client_id}: 클래스 0 정밀도: {precision_0:.4f}, 재현율: {recall_0:.4f}, F1: {f1_0:.4f}")
        print(f"클라이언트 {self.client_id}: 클래스 1 정밀도: {precision_1:.4f}, 재현율: {recall_1:.4f}, F1: {f1_1:.4f}")

        # 반환 값에 각 클래스별 지표 추가
        return float(accuracy), len(self.X_val), {
            "precision_0": float(precision_0), "recall_0": float(recall_0), "f1_0": float(f1_0),
            "precision_1": float(precision_1), "recall_1": float(recall_1), "f1_1": float(f1_1)
            }

def start_client(client_id: int):
    """Flower 클라이언트를 시작합니다."""
    print(f"Flower 클라이언트 {client_id} 시작 중...")
    # Flower 서버에 연결합니다.
    fl.client.start_client(server_address="127.0.0.1:8081", client=FlowerClient(client_id).to_client())
    print(f"클라이언트 {client_id} 중지됨.")

if __name__ == "__main__":
    # 명령줄 인수로 CLIENT_ID가 제공되었는지 확인합니다.
    if len(sys.argv) > 1:
        try:
            CLIENT_ID = int(sys.argv[1])
        except ValueError:
            print("오류: CLIENT_ID는 정수여야 합니다.")
            sys.exit(1)
    else:
        print("사용법: python client.py <CLIENT_ID>")
        print("예시: python client.py 1")
        sys.exit(1)

    start_client(CLIENT_ID)

