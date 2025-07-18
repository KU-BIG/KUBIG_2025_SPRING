import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from model.HetDDI import HetDDI
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test
from utils.pytorchtools import EarlyStopping
from utils.logger import Logger
import os
from tdc.utils import get_label_map   # DrugBank interaction 라벨 정보 불러오기
from easydict import EasyDict

def run(args):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    data_path = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name)
    
    # 저장 경로 설정
    ckpt_root = os.path.join("checkpoints", args.ddi_name, args.label_type, args.mode, args.condition)
    os.makedirs(ckpt_root, exist_ok=True)
    # kg_g & smiles 캐시 사용
    kg_cache_path = os.path.join(ckpt_root, "kg_data.pt")
    if os.path.exists(kg_cache_path):
        kg_data = torch.load(kg_cache_path)
        kg_g, smiles = kg_data["kg_g"], kg_data["smiles"]
    else:
        kg_g, smiles = load_data(data_path, device=device)
        torch.save({"kg_g": kg_g, "smiles": smiles}, kg_cache_path)
    
    # interaction label id와 interaction label text 매핑
    if args.label_type == 'multi_class':
        id2label = get_label_map(name='DrugBank', task='DDI')
    elif args.label_type == 'multi_label':
        id2label = get_label_map(name='TWOSIDES', task='DDI', name_column='Side Effect Name')
    else:
        id2label = None
        
    train_sample, test_sample = get_train_test(data_path, fold_num=args.fold_num,
                                               label_type=args.label_type, condition=args.condition)

    scores = []
    for i in range(0, args.fold_num):
        # for i in range(0, 1):
        fold_dir = os.path.join(ckpt_root, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # load data
        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2:]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2:]

        if args.label_type == 'multi_class':
            train_y = torch.from_numpy(train_y).long()
            test_y = torch.from_numpy(test_y).long()
        else:
            train_y = torch.from_numpy(train_y).float()
            test_y = torch.from_numpy(test_y).float()

        # load model
        if args.label_type == 'multi_class':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 86, args.condition).to(device)
            loss_func = nn.CrossEntropyLoss()
        elif args.label_type == 'binary_class':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 1, args.condition).to(device)
            loss_func = nn.BCEWithLogitsLoss()
        elif args.label_type == 'multi_label':
            model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 200, args.condition).to(device)
            loss_func = nn.BCEWithLogitsLoss()
        if i == 0:
            print(model)
            
        # 모델 초기 저장 (초기화된 모델의 가중치/파라미터 상태, 학습 시작 직전의 모델 상태 저장)
        torch.save(model.state_dict(), os.path.join(fold_dir, "model_init.pt"))


        # divide parameters into two parts, weight_p has l2_norm but bias_bn_emb_p not
        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name or 'bn' in name or 'embedding' in name:
                bias_bn_emb_p += [p]
            else:
                weight_p += [p]
        model_parameters = [
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_bn_emb_p, 'weight_decay': 0},
        ]
        # train setting
        optimizer = optim.Adam(model_parameters, lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        best_test_score = None
        for epoch in range(args.epoch):
            train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                            i, epoch, args.batch_size, args.label_type, device)

            test_score = test(model, loss_func, test_x_left, test_x_right, test_y, i, epoch, args.batch_size,
                              args.label_type, device)

            test_acc = test_score[0]
            if epoch > 50:
                early_stopping(test_acc, model)
                if early_stopping.counter == 0:
                    best_test_score = test_score
                    torch.save(model.state_dict(), os.path.join(fold_dir, "best_model.pt"))
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break
            # one epoch end
            print(best_test_score)
            print("=" * 100)
            
        # fold 결과 저장
        result_path = os.path.join(fold_dir, "result.npy")
        np.save(result_path, best_test_score)
        
        # 예측 결과 저장 (multi_class 전용)
        if args.label_type == 'multi_class' and id2label:
            # test time
            # 모델 로드 및 예측
            model.load_state_dict(torch.load(os.path.join(fold_dir, "best_model.pt")))
            model.eval()
            with torch.no_grad():
                pred_logits = model(test_x_left.to(device), test_x_right.to(device))
                pred_ids = torch.argmax(pred_logits, dim=1).cpu().numpy()
                
        # one fold end
        scores.append(best_test_score)
        print('Test set score:', scores)
        
    # all fold end, output the final result
    # 각 폴드의 acuuracy 비교 -> 가장 높은 정확도의 checkpoint 불러오기
    best_fold_idx = -1
    best_acc = -1
    best_model_path = None

    for i in range(args.fold_num):
        result_path = os.path.join(ckpt_root, f"fold_{i}", "result.npy")
        if os.path.exists(result_path):
            result = np.load(result_path)
            acc = result[0]  # accuracy 첫 번째 인덱스
            if acc > best_acc:
                best_acc = acc
                best_fold_idx = i
                best_model_path = os.path.join(ckpt_root, f"fold_{i}", "best_model.pt")

    print(f"Best fold is {best_fold_idx} with accuracy {best_acc:.4f}")

    # best model의 input/output만 다시 예측해서 CSV로 저장 (multi_class 한정)
    if args.label_type == 'multi_class' and best_model_path and id2label:
        test_x_left = test_sample[best_fold_idx][:, 0]
        test_x_right = test_sample[best_fold_idx][:, 1]
        test_y = test_sample[best_fold_idx][:, 2:]

        test_y = torch.from_numpy(test_y).long()

        model = HetDDI(kg_g, smiles, args.hidden_dim, args.num_layer, args.mode, 86, args.condition).to(device)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        with torch.no_grad():
            pred_logits = model(test_x_left.to(device), test_x_right.to(device))
            pred_ids = torch.argmax(pred_logits, dim=1).cpu().numpy()

        ## 결과물 출력 및 csv 저장 부분 ##
        pred_records = []
        for j in range(len(test_x_left)):
            left_idx = int(test_x_left[j])
            right_idx = int(test_x_right[j])
            label_id = int(pred_ids[j])
            label_text = id2label.get(label_id, "UNKNOWN")
            pred_records.append({
                "left_idx": left_idx,
                "right_idx": right_idx,
                "predicted_label_id": label_id,
                "predicted_label_text": label_text
            })

        pred_df = pd.DataFrame(pred_records)
        pred_df.to_csv(os.path.join(ckpt_root, "best_model_predictions.csv"), index=False)
        print(f"Saved best model predictions to {os.path.join(ckpt_root, 'best_model_predictions.csv')}")
        ## 결과물 출력 및 csv 저장 부분 ##

    scores = np.array(scores)
    scores = scores.mean(axis=0)
    if args.label_type == 'multi_class':
        mean_kappa = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.03f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, kappa:{:.3f}\033[0m"
            .format(
                scores[0], scores[1], scores[2], scores[3], scores[4]
            ))
    elif args.label_type == 'binary_class':
        mean_auc = scores[:, 4].mean()
        print(
            "\033[1;31mFinal DDI result:\n"
            "acc:{:.3f}, f1:{:.3f}, precision:{:.3f}, recall:{:.3f}, auc:{:.3f}\033[0m"
            .format(
                scores[0], scores[1], scores[2], scores[3], scores[4]
            ))
    elif args.label_type == 'multi_label':
        print(scores)



if __name__ == '__main__':
    # ap = argparse.ArgumentParser(description='')
    # ap.add_argument('--batch_size', type=int, default=2 ** 15)
    # ap.add_argument('--fold_num', type=int, default=5)
    # ap.add_argument('--hidden_dim', type=int, default=300, help='Dimension of the node hidden state. Default is 300.')
    # ap.add_argument('--num_layer', type=int, default=3)
    # ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    # ap.add_argument('--patience', type=int, default=50)
    # ap.add_argument('--lr', type=float, default=1e-3)
    # ap.add_argument('--weight_decay', type=float, default=1e-5)

    # ap.add_argument('--label_type', type=str, choices=['multi_class', 'binary_class', 'multi_label'],
    #                 default='binary_class')
    # ap.add_argument('--condition', type=str, choices=['s1', 's2', 's3'], default='s1')
    # ap.add_argument('--mode', type=str, choices=['only_kg', 'only_mol', 'concat'], default='concat')
    # ap.add_argument('--data_path', type=str, default='./data')
    # ap.add_argument('--kg_name', type=str, default='DRKG')
    # ap.add_argument('--ddi_name', type=str, choices=['DrugBank', "TWOSIDES"], default='DrugBank')

    # args = ap.parse_args(args=[])
    # print(args)

    # terminal = sys.stdout
    # log_file = './log/ddi-dataset_{} label-type_{} mode_{} condition_{}.txt'. \
    #     format(args.hidden_dim, args.label_type, args.mode,args.condition)
    # sys.stdout = Logger(log_file, terminal)

    # import warnings
    # warnings.filterwarnings("ignore", category=UserWarning)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # # device = torch.device('cpu')
    # print('running on', device)

    # run(args)

    args = EasyDict({
        'batch_size': 2 ** 15,
        'fold_num': 5,
        'hidden_dim': 300,
        'num_layer': 3,
        'epoch': 100,
        'patience': 50,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'label_type': 'multi_class',  # binary_class
        'condition': 's1',
        'mode': 'concat',
        
        'data_path': './data',
        'kg_name': 'FOODRKG',
        'ddi_name': 'DrugBank',
    })

    # print 출력을 터미널과 로그 파일에 동시에 기록하도록 설정 & warning 무시
    sys.stdout = Logger(
        f'./log/ddi-dataset_{args.hidden_dim} '
        f'label-type_{args.label_type} '
        f'mode_{args.mode} '
        f'condition_{args.condition}.txt',
        sys.stdout
    )
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); print('running on', device)

    run(args)
