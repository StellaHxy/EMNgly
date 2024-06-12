from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef,roc_auc_score, confusion_matrix
import pickle as pk
import pandas as pd
import numpy as np
import sys
import pickle 
import argparse
import os
import time
import torch
import tqdm


def get_features(data_csv, dataset_dir):
    ret_x = []
    ret_y = []
    df = pd.read_csv(data_csv)
    for index, row in tqdm.tqdm(df.iterrows()):
        features = []
        try:
            label = int(row['label'])
            id = row['id']
            pos = int(row['pos'])
            for key in ['site', 'local', 'structure']:
                if key != 'structure':
                    pkl_file = os.path.join(dataset_dir, key, f'{index}.pkl')
                else:
                    pkl_file = os.path.join(dataset_dir, key, f'{id}.pkl')
                pklf = open(pkl_file,'rb')
                emb_key = str(key)+'_emb'
                if key != 'structure':
                    data = pk.load(pklf)[emb_key]
                else:
                    data = pk.load(pklf)[emb_key][pos-1]
                features.append(data)
        except: 
            continue
        featureAll = torch.cat(features, dim=0)
        featureAll = featureAll.detach().numpy()
        ret_x.append(featureAll)
        ret_y.append(label)
    return np.array(ret_x), np.array(ret_y)


def model(test_x, ckpt_path):
    with open(ckpt_path,'rb') as f:  
        model = pickle.load(f)

    predict_y_2 = model.predict_proba(test_x)
    predict_y = predict_y_2[:,-1]
    return predict_y.reshape(1,len(test_x))


def get_scores(label_y, predict_y, th=0.5):
    ret_dict = {}

    auc = roc_auc_score(label_y, predict_y)
    ret_dict['auc'] = round(auc,5)
    predict_label = np.where(np.array(predict_y) > th, 1, 0)

    mcc = matthews_corrcoef(label_y, predict_label)
    ret_dict['mcc'] = round(mcc,5)

    # predict_label = np.array(predict_label)
    acc = accuracy_score(label_y, predict_label)
    ret_dict['acc'] = round(acc,5)

    tn, fp, fn, tp = confusion_matrix(label_y, predict_label).ravel()
    
    specificity = tn / (tn + fp)
    ret_dict['specificity'] = round(specificity, 5)
    
    recall = recall_score(label_y, predict_label)
    ret_dict['sensitivity'] = round(recall, 5)

    return ret_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--data_path', type=str, default="./data/N-GlycositeAltas/")
    parser.add_argument('--ckpt_path', type=str, default="checkpoints/N-GlyAltas_classifier.pkl")
    parser.add_argument('--log_dir', type=str, default="./log/")
    args = parser.parse_args()

    input_csv = os.path.join(args.data_path, args.mode+".csv")
    emb_dir = os.path.join(args.data_path, "features" ,args.mode)
    ckpt_path = args.ckpt_path
    
    print("start reading test data...")
    
    test_X, test_Y = get_features(input_csv, emb_dir)

    print("test data count:", len(test_X), len(test_Y))
    predict_y = model(test_X, ckpt_path)
    predict_y = predict_y.tolist()
    scores = get_scores(np.array(test_Y), predict_y[0])
    
    time_ = time.time()
    os.makedirs(args.log_dir, exist_ok=True)
    with open(f'./log/score_{time_}.log', 'w') as f:
        f.write(str(scores))

    print(scores)
    