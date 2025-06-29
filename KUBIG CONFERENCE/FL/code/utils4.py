import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# 모든 데이터셋에서 제거할 변수 목록 정의
# 이 목록에 있는 변수가 데이터셋에 없으면 무시됩니다.
COLUMNS_TO_REMOVE = [
    'Transaction_Resumed_Update',
    'Transaction_Resumed_Update_count',
    'Transaction_Resumed_Update_count_0',
    'Account_transaction_count_index',
    'First_transaction_label',
    'Only_Sender_not_recipient',
    'Only_Sender_not_recipient_single_transaction'
]

def load_combined_validation_data_for_server_eval(client_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    지정된 파티션들로부터 검증 데이터셋을 로드하고 결합하여 서버 측 중앙 집중식 테스트 세트로 사용합니다.
    (수정됨: 지정된 변수들을 제거하는 로직 추가)
    """
    all_X_val = []
    all_y_val = []
    for client_id in client_ids:
        try:
            X_val_path = f'partition{client_id}_X_val.csv'
            y_val_binary_path = f'partition{client_id}_y_val_binary.csv'

            # X 데이터 로드
            X_val_df = pd.read_csv(X_val_path, header=0)
            # Y 데이터 로드
            y_val_df = pd.read_csv(y_val_binary_path, header=0)

            # [수정된 부분]: X 데이터에서 지정된 열 제거
            # errors='ignore'를 사용하여 열이 없으면 오류 없이 건너뜁니다.
            cols_to_drop_existing_in_X = [col for col in COLUMNS_TO_REMOVE if col in X_val_df.columns]
            if cols_to_drop_existing_in_X:
                X_val_df = X_val_df.drop(columns=cols_to_drop_existing_in_X)
                print(f"서버: 파티션 {client_id} X_val에서 {cols_to_drop_existing_in_X} 열 제거 완료.")

            # Y 데이터는 일반적으로 단일 열이므로 제거할 변수가 없을 것으로 예상되지만,
            # 만약을 대비해 확인하는 것이 안전합니다.
            cols_to_drop_existing_in_y = [col for col in COLUMNS_TO_REMOVE if col in y_val_df.columns]
            if cols_to_drop_existing_in_y:
                y_val_df = y_val_df.drop(columns=cols_to_drop_existing_in_y)
                print(f"서버: 파티션 {client_id} y_val에서 {cols_to_drop_existing_in_y} 열 제거 완료.")
            
            X_val = X_val_df.values
            y_val = y_val_df.values.ravel() # 1D 배열을 위해 .ravel() 사용

            all_X_val.append(X_val)
            all_y_val.append(y_val)
            print(f"서버: 파티션 {client_id}로부터 검증 데이터 로드됨 - X_val 형태: {X_val.shape}, y_val 형태: {y_val.shape}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"파티션 {client_id}의 검증 데이터 파일이 올바른 디렉토리에 있는지 확인하십시오. {e}")
        except Exception as e:
            raise Exception(f"파티션 {client_id}의 검증 데이터 로드 오류: {e}")
    
    X_combined_test = np.vstack(all_X_val)
    y_combined_test = np.hstack(all_y_val)

    return X_combined_test, y_combined_test

def load_partition_train_data(client_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    특정 파티션의 훈련 데이터셋을 로드합니다.
    (수정됨: 지정된 변수들을 제거하는 로직 추가)
    """
    try:
        X_train_path = f'partition{client_id}_X_train.csv'
        y_train_binary_path = f'partition{client_id}_y_train_binary.csv'

        # X 데이터 로드
        X_train_df = pd.read_csv(X_train_path, header=0)
        # Y 데이터 로드
        y_train_df = pd.read_csv(y_train_binary_path, header=0)

        # [수정된 부분]: X 데이터에서 지정된 열 제거
        cols_to_drop_existing_in_X = [col for col in COLUMNS_TO_REMOVE if col in X_train_df.columns]
        if cols_to_drop_existing_in_X:
            X_train_df = X_train_df.drop(columns=cols_to_drop_existing_in_X)
            print(f"서버/클라이언트: 파티션 {client_id} X_train에서 {cols_to_drop_existing_in_X} 열 제거 완료.")

        cols_to_drop_existing_in_y = [col for col in COLUMNS_TO_REMOVE if col in y_train_df.columns]
        if cols_to_drop_existing_in_y:
            y_train_df = y_train_df.drop(columns=cols_to_drop_existing_in_y)
            print(f"서버/클라이언트: 파티션 {client_id} y_train에서 {cols_to_drop_existing_in_y} 열 제거 완료.")

        X_train = X_train_df.values
        y_train = y_train_df.values.ravel()

        print(f"서버: 파티션 {client_id}로부터 훈련 데이터 로드됨 - X_train 형태: {X_train.shape}, y_train 형태: {y_train.shape}")
        return X_train, y_train
    except FileNotFoundError as e:
        raise FileNotFoundError(f"파티션 {client_id}의 훈련 데이터 파일이 올바른 디렉토리에 있는지 확인하십시오. {e}")
    except Exception as e:
        raise Exception(f"파티션 {client_id}의 훈련 데이터 로드 오류: {e}")

def load_client_data(client_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    특정 클라이언트(파티션)의 전처리된 데이터(train, valid ,test)를 로드합니다.
    (수정됨: 지정된 변수들을 제거하는 로직 추가)
    
    """
    try:
        # X_train, y_train_binary 로드
        Xn_train_df = pd.read_csv(f'partition{client_id}_X_train.csv', header=0)
        yn_train_binary_df = pd.read_csv(f'partition{client_id}_y_train_binary.csv', header=0)

        # X_val, y_val_binary 로드
        Xn_val_df = pd.read_csv(f'partition{client_id}_X_val.csv', header=0)
        yn_val_binary_df = pd.read_csv(f'partition{client_id}_y_val_binary.csv', header=0)
        
        # X_test 로드
        Xn_test_df = pd.read_csv(f'partition{client_id}_X_test.csv', header=0)
        
        # [수정된 부분]: 각 데이터프레임에서 지정된 열 제거
        datasets = {
            'Xn_train': Xn_train_df, 'yn_train_binary': yn_train_binary_df,
            'Xn_val': Xn_val_df, 'yn_val_binary': yn_val_binary_df,
            'Xn_test': Xn_test_df
        }

        for name, df in datasets.items():
            cols_to_drop_existing = [col for col in COLUMNS_TO_REMOVE if col in df.columns]
            if cols_to_drop_existing:
                datasets[name] = df.drop(columns=cols_to_drop_existing)
                print(f"클라이언트 {client_id}: {name}에서 {cols_to_drop_existing} 열 제거 완료.")

        Xn_train = datasets['Xn_train'].values
        yn_train_binary = datasets['yn_train_binary'].values.ravel()

        Xn_val = datasets['Xn_val'].values
        yn_val_binary = datasets['yn_val_binary'].values.ravel()
        
        Xn_test = datasets['Xn_test'].values
        
        print(f"클라이언트 {client_id}: 로드된 X_train 형태: {Xn_train.shape}, y_train 형태: {yn_train_binary.shape}")
        print(f"클라이언트 {client_id}: 로드된 X_val 형태: {Xn_val.shape}, y_val 형태: {yn_val_binary.shape}")
        print(f"클라이언트 {client_id}: 로드된 X_test 형태: {Xn_test.shape}")
        print(f"클라이언트 {client_id}: X_train에는 {Xn_train.shape[1]}개의 특성이 있습니다.")
        print(f"클라이언트 {client_id}: X_val에는 {Xn_val.shape[1]}개의 특성이 있습니다.")
        print(f"클라이언트 {client_id}: X_test에는 {Xn_test.shape[1]}개의 특성이 있습니다.")

        return Xn_train, Xn_val, Xn_test, yn_train_binary, yn_val_binary
    except FileNotFoundError as e:
        raise FileNotFoundError(f"파티션 {client_id}의 데이터 파일이 올바른 디렉토리에 있는지 확인하십시오. {e}")
    except Exception as e:
        raise Exception(f"파티션 {client_id}의 데이터 로드 오류: {e}")

