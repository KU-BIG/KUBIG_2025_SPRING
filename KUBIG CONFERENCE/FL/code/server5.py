import flwr as fl
import numpy as np
from sklearn.linear_model import SGDClassifier # LogisticRegression 대신 SGDClassifier 임포트
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , classification_report # classification_report 추가
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from typing import Dict, Optional, Tuple, List
import utils4 as utils # 유틸리티 함수 임포트

# client.py와 동일한 하이퍼파라미터로 기본 모델을 정의합니다.
# 이 모델들은 서버에서 메타 특성 생성에 사용됩니다.
def create_base_models_for_server():
    xgb_model = XGBClassifier(
        n_estimators=455,
        max_depth=6,
        learning_rate=0.08653415292886991,
        subsample=0.8425998022058797,
        colsample_bytree=0.8996549142330391,
        random_state=42,
    )
    cat_model = CatBoostClassifier(
        iterations=520,
        depth=6,
        learning_rate=0.09980327654097068,
        l2_leaf_reg=4.545010679457599,
        random_seed=42,
        verbose=0, # 학습 중 상세 출력을 억제합니다.
    )
    return xgb_model, cat_model


# 서버 측 평가를 수행하는 함수를 반환합니다.
# 이 함수는 전략의 `evaluate_fn`에 전달됩니다.
# 전역 SGDClassifier 메타 학습자를 평가합니다. (변경됨: LogisticRegression -> SGDClassifier)
def get_on_evaluate_fn(centralized_testset: Tuple[np.ndarray, np.ndarray], partition4_trainset: Tuple[np.ndarray, np.ndarray]):
    X_combined_test, y_combined_test = centralized_testset
    X_p4_train, y_p4_train = partition4_trainset
    
    # FL 라운드가 시작되기 전에 서버 측 기본 모델을 한 번 초기화하고 훈련합니다. (파티션 4 훈련 데이터를 사용)
    # 이 모델들은 서버의 평가 및 메타 학습자 초기화를 위한 고정 특성 추출기 역할을 합니다.
    print("서버: 메타 특성 생성을 위해 파티션 4 훈련 데이터를 사용하여 서버 측 기본 모델 초기화 및 훈련...")
    server_base_xgb, server_base_cat = create_base_models_for_server()

    # 파티션 4의 훈련 데이터로 기본 모델을 훈련합니다.
    # 이는 메타 학습자를 위한 메타 특성을 생성할 준비를 합니다.
    # 기본 모델이 올바르게 훈련되도록 대상에 최소 두 개의 클래스가 있는지 확인합니다.
    if len(np.unique(y_p4_train)) < 2:
        raise ValueError("파티션 4 훈련 데이터에 기본 모델 훈련에 필요한 2개 미만의 클래스가 있습니다.")
    
    server_base_xgb.fit(X_p4_train, y_p4_train)
    server_base_cat.fit(X_p4_train, y_p4_train)
    print("서버: 파티션 4 훈련 데이터를 사용하여 서버 측 기본 모델 훈련 완료.")
          
    def evaluate(server_round: int, parameters: List[fl.common.NDArray], config: Dict) -> Optional[Tuple[float, Dict]]:
        """
        중앙 집중식 테스트 세트에서 전역 SGDClassifier 메타 학습자를 평가합니다.
        (변경됨: LogisticRegression -> SGDClassifier)
        """        
        
        # 평가를 위해 SGDClassifier 모델을 다시 초기화합니다.
        # 로지스틱 회귀를 위한 SGDClassifier를 사용합니다. loss='log_loss'
        model = SGDClassifier(loss='log_loss', random_state=42, warm_start=True, eta0 = 0.05, learning_rate='adaptive') 
        # warm_start=True: 이전 fit 호출의 솔루션을 재사용하여 추가 훈련이 가능하게 합니다.
        # 그러나 여기서는 `fit`이 아니라 `coef_`와 `intercept_`를 직접 설정하므로 주로 `classes_` 초기화를 위해 사용됩니다.


        # 훈련된 서버 측 기본 모델을 사용하여 서버의 결합된 테스트 세트에 대한 메타 특성을 생성합니다.
        meta_features_xgb = server_base_xgb.predict_proba(X_combined_test)
        meta_features_cat = server_base_cat.predict_proba(X_combined_test)
        
        # 메타 특성을 연결합니다. 이진 분류의 경우 (샘플 수, 4)가 됩니다.
        X_meta_features_for_server_eval = np.hstack([meta_features_xgb, meta_features_cat])

        print(f"서버: 서버 평가 시작 시 X_combined_test 형태: {X_combined_test.shape}")
        print(f"서버: 서버 평가를 위해 생성된 메타 특성 형태: {X_meta_features_for_server_eval.shape}")
        
        # 중요: SGDClassifier 모델을 더미 데이터에 맞게 훈련하여 'classes_' 속성을 초기화하고
        # 메타 학습자(예: 4)에 대한 coef_의 특성 수가 올바른지 확인합니다.
        num_meta_features_expected = X_meta_features_for_server_eval.shape[1]
        
        # SGDClassifier는 `partial_fit` 또는 `fit`을 통해 `classes_`를 초기화해야 합니다.
        # 여기서는 `partial_fit`을 사용하여 `classes_`와 `n_features_in_`를 설정합니다.
        dummy_X_sgd = np.array([[0.0] * num_meta_features_expected, [1.0] * num_meta_features_expected])
        dummy_y_sgd = np.array([0, 1])
        model.partial_fit(dummy_X_sgd, dummy_y_sgd, classes=np.array([0, 1])) # classes 명시적으로 지정
        # `classes` 매개변수는 `partial_fit`의 첫 호출에서 지정되어야 합니다.

        # 연합 클라이언트로부터 받은 전역 SGDClassifier 메타 학습자 매개변수를 설정합니다.
        model.coef_ = parameters[0]
        model.intercept_ = parameters[1]

        print(f"서버: 클라이언트 매개변수 설정 후 모델 coef_ 형태: {model.coef_.shape}")
        print(f"서버 : 클라이언트로 부터 받은 매게변수의 {server_round}번째 AVG값 : \n coef : {parameters[0]}, \n intercept : {parameters[1]}")
       
        # 생성된 메타 특성으로 예측을 수행합니다.
        y_pred = model.predict(X_meta_features_for_server_eval)

         # 평가 지표 계산 및 각 클래스별 지표 포함 (수정된 부분)
        accuracy = accuracy_score(y_combined_test, y_pred)
        
        # classification_report를 사용하여 각 클래스별 정밀도, 재현율, F1-점수 추출
        report = classification_report(y_combined_test, y_pred, output_dict=True, zero_division=0)

        precision_0 = report['0']['precision']
        recall_0 = report['0']['recall']
        f1_0 = report['0']['f1-score']

        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_1 = report['1']['f1-score']
        
        # 전체 가중 평균 값도 유지 (선택 사항이지만 유용할 수 있음)
        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']
        weighted_f1 = report['weighted avg']['f1-score']


        print(f"\n서버 라운드 {server_round} 중앙 집중식 평가:")
        print(f"  정확도: {accuracy:.4f}")
        print(f"  클래스 0 정밀도: {precision_0:.4f}, 재현율: {recall_0:.4f}, F1: {f1_0:.4f}")
        print(f"  클래스 1 정밀도: {precision_1:.4f}, 재현율: {recall_1:.4f}, F1: {f1_1:.4f}")
        print(f"  (가중 평균) 정밀도: {weighted_precision:.4f}, 재현율: {weighted_recall:.4f}, F1: {weighted_f1:.4f}\n")

        # 반환 값에 각 클래스별 지표 추가
        return float(accuracy), {
            "precision_0": float(precision_0), "recall_0": float(recall_0), "f1_0": float(f1_0),
            "precision_1": float(precision_1), "recall_1": float(recall_1), "f1_1": float(f1_1),
            "weighted_precision": float(weighted_precision), # 가중 평균도 필요하면 추가
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1)
        }

    return evaluate


def get_initial_parameters(partition4_trainset: Tuple[np.ndarray, np.ndarray]):
    """
    서버의 메타 학습자(SGDClassifier)의 초기 매개변수를 아무 숫자로만 초기화합니다.
    (수정된 부분: 파티션 4 훈련 데이터를 사용하여 학습하지 않고, 랜덤 값으로 초기화)
    """
    # X_p4_train, y_p4_train = partition4_trainset # 이제 이 데이터는 초기화에 사용되지 않습니다.

    # 메타 학습자의 특성 수는 2개의 기본 모델 * 각 2개의 확률(이진 분류) = 4개 입니다.
    num_meta_features = 4 
    
    # SGDClassifier 모델을 정의하고 coef_와 intercept_를 임의의 값으로 초기화합니다.
    # 이 초기화는 특정 데이터 학습에 의존하지 않고, 무작위 값에서 시작합니다.
    # coef_는 (클래스 수, 특성 수) 형태, intercept_는 (클래스 수,) 형태를 가집니다.
    # 이진 분류의 경우 (1, num_meta_features)와 (1,)이 됩니다.
    initial_coef = np.random.rand(1, num_meta_features) * 0.1 - 0.05 # 작은 랜덤 값으로 초기화
    initial_intercept = np.random.rand(1) * 0.1 - 0.05

    print(f"서버: 메타 학습자 초기 매개변수 랜덤 값으로 설정 완료 - coef_ 형태: {initial_coef.shape}, intercept_ 형태: {initial_intercept.shape}")

    # 계수와 절편을 초기 매개변수로 반환합니다.
    return [initial_coef, initial_intercept]



def main():
    # 1. 서버를 위한 중앙 집중식 테스트 데이터셋을 준비합니다.
    # 이는 파티션 1, 2, 3, 4의 검증 데이터를 결합하여 중앙 집중식 테스트 세트를 구성합니다.
    print("서버: 파티션 1, 2, 3, 4의 검증 세트로부터 중앙 집중식 테스트 데이터 로드 중...")
    centralized_testset = utils.load_combined_validation_data_for_server_eval(client_ids=[1, 2, 3, 4])
    print(f"서버 평가를 위한 결합된 테스트 데이터 형태: X: {centralized_testset[0].shape}, y: {centralized_testset[1].shape}")

    # 2. 서버의 기본 모델 훈련 및 메타 학습자 초기화를 위해 파티션 4의 훈련 데이터를 로드합니다.
    print("서버: 기본 모델 훈련 및 메타 학습자 초기화를 위해 파티션 4의 훈련 데이터 로드 중...")
    partition4_trainset = utils.load_partition_train_data(client_id=4)
    print(f"기본 모델 훈련을 위한 파티션 4 훈련 데이터 형태: X: {partition4_trainset[0].shape}, y: {partition4_trainset[1].shape}")
    
        # 클라이언트 훈련 라운드마다 설정될 config를 반환하는 함수
    def fit_config_fn(server_round: int) -> Dict:
        """Return a configuration dict for the current round."""
        return {"local_epochs": 10} # 클라이언트가 라운드 당 몇 에포크를 훈련하도록 설정
    
    # 3. 연합 학습 전략(FedAvg)을 정의합니다.
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # 모든 클라이언트(4 클라이언트)가 훈련에 참여합니다.
        min_fit_clients=4,          # 훈련을 시작하는 데 필요한 최소 4 클라이언트입니다.
        fraction_evaluate=1.0,      # 모든 클라이언트(4 클라이언트)가 평가에 참여합니다.
        min_evaluate_clients=4,     # 평가를 시작하는 데 필요한 최소 4 클라이언트입니다.
        min_available_clients=4,    # 연합 학습을 시작하는 데 필요한 최소 4 클라이언트입니다.
        
        # 결합된 검증 데이터를 테스트 세트로 사용하는 중앙 집중식 평가 함수입니다.
        evaluate_fn=get_on_evaluate_fn(centralized_testset, partition4_trainset), 
        
        # 파티션 4의 메타 특성으로 훈련된 매개변수로 전역 모델(메타 학습자)을 초기화합니다.
        initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters(partition4_trainset)),

        # local_epochs를 설정하여 각 클라이언트에서의 SGDClassifier 훈련 에폭 수를 제어합니다.
        on_fit_config_fn= fit_config_fn, # 클라이언트 훈련 라운드마다 설정될 config를 반환하는 함수
    )

    # 4. Flower 서버를 시작합니다.
    fl.server.start_server(
        server_address="0.0.0.0:8081", # 모든 인터페이스에서 수신합니다.
        config=fl.server.ServerConfig(num_rounds=5), # SGDClassifier를 위한 15 라운드입니다.
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

