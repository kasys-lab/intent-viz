import json
from sys import modules
from typing import AsyncIterable
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from models import BiAttention, SingleColBertWithFeature
from feature_extraction.features import extract_column_features

data_file = "./data/input_data.json"
embed_path = "./data/glove.6B.100d.txt"
type_model_path = "./saved_models/bi_att_for_col_model.pt"
column_model_path = "./saved_models/single_bert_fea2.pt"
remove_feature_num = [11, 12, 47, 49, 51]
Types = {
            0: "Area",
            1: "Bar",
            2: "Circle",
            3: "Line",
            4: "Multi polygon",
            5: "Pie",
            6: "Shape",
            7: "Square"
        }



def get_processed_feature(features):
    scaler = StandardScaler()
    processed_feature = []
    bool_column = []
    train_feature2 = np.array(features)
    train_features2 = train_feature2.T
    
    train_features2 = train_features2.astype(np.float32)
    for idx, feature1 in enumerate(train_features2):
        if np.isnan(np.nanmean(feature1)):
            feature1 = np.zeros(len(feature1))
        if idx in remove_feature_num:
            continue
        #boolean
        if idx in bool_column:
            uniqs, counts = np.unique(feature1, return_counts=True)
            feature1[np.isnan(feature1)] = uniqs[counts == np.amax(counts)].min()
        #numeric
        else:
            #exist inf
            if np.isinf(feature1).any():
                inf_bool = np.isinf(feature1)
                noinf_feature1 = feature1[np.logical_not(inf_bool)]
                feature1[inf_bool] = np.nanmax(noinf_feature1) 
                feature1 = np.clip(feature1, np.nanpercentile(feature1, 1), np.nanpercentile(feature1, 99))
            if np.nanmax(feature1) != 0:
                feature1 = feature1 / np.nanmax(feature1)
            feature1[np.isnan(feature1)] = np.nanmean(feature1)
        
        feature2 = scaler.fit_transform(feature1.reshape(-1, 1))
        processed_feature.append(feature2)
    processed_feature = np.array(processed_feature).T
    return processed_feature

def get_vec_dict(path):
    vec_dict = {}
    with open(path) as f:
        for line in f:
            line = line.split()
            vec_dict[line[0]] = np.array(list(map(float, line[1:]))) 
    return vec_dict

def main():

    remove_list = {".", ",", "(", ")", "[", "]", "{", "}", "=", "!", "?", "*", ' ', '\t', ";", ":"}
    emb = get_vec_dict(embed_path)

    #### 入力データを配列に変換
    with open(data_file, "r") as f:
        jsn = json.load(f)
        visualization_intent = jsn["visualization_intent"]
        data = []    
        for key in jsn["data"][0].keys():
            one_data = [key]
            for value in jsn["data"]:       
                one_data.append(value[key])
            data.append(one_data)

    #### 特徴抽出 & 標準化
    ### fid, field_id, moment_5, moment_7, moment_9
    column_features_with_use = extract_column_features(data, 0, 0)
    feature = []
    headers = []
    for header, features in column_features_with_use.items():
        features[12] = (0, 0)
        feature.append([i[1] for i in features])
        headers.append(header)
    processed_features = get_processed_feature(feature)
        

    #### 単語埋め込み
    embed_data = []
    for header, feature in zip(headers, processed_features[0]):
        head_vec = np.array([0]*100)
        header_words = header.split()
        lang_count = 0
        for header_word in header_words:
            lang_count += 1
            header_word = header_word.lower()
            if not header_word in remove_list:
                if header_word in emb:
                    head_vec = head_vec + emb[header_word]
                else:
                    head_vec = head_vec + np.array([0]*100)

        if len(header_words) != 0:
            head_vec = head_vec / lang_count
        embed_data.append(list(head_vec) + list(feature))
    if len(embed_data) < 31:
        embed_data = list(embed_data) + [[0]*(len(feature)+100)] * (30 - len(embed_data))
    embed_data = torch.Tensor(embed_data)


    embed_title = []
    words = visualization_intent.split()
    for idx, word in enumerate(words):
        word = word.lower()
        if idx == 12:
            break
        if not word in remove_list:
            if word in emb:
                embed_title.append(emb[word])
        
    if len(embed_title) < 13:
        embed_title = embed_title + [[0]*100] * (12 - len(embed_title))
    embed_title = torch.Tensor(embed_title)

    #### モデルに入力

    #### 視覚化種類の予測
    '''
    model input 
    [
        [
            [[word1 vecor], [word2 vector], [word3 vector], ...],
            [[header1 mean vector, statistical features1], [header mean vector, statistical features2], ...]
        ],[
            [[word1 vecor], [word2 vector], [word3 vector], ...],
            [[header1 mean vector, statistical features1], [header mean vector, statistical features2], ...]
        ]
    ]
    '''
    type_model = BiAttention()
    type_model.load_state_dict(torch.load(type_model_path))
    type_model.eval()
    output = type_model([embed_title, embed_data])
    predict = torch.max(output.data, 1)[1]

    #### 視覚化列の予測
    '''
    model input 
    title_single_col : titleとheaderのペアのtokenizerの出力(列は30列にpadding)
    column_feature : [
                        [statistical features 1],
                        [statistical deatures 2], ...
                    ]    
    '''
    processed_features = processed_features[0].tolist()
    if len(processed_features) > 30:
        processed_features = processed_features[:30]
        headers = headers[:30]
    elif len(processed_features) != 30:
        processed_features+=[[0]*78]*(30-len(processed_features))
        headers+=[""]*(30-len(headers))
    headers = [[i] for i in headers]
    titles = []
    for idx, feature in enumerate(processed_features):
        if feature != [0]*78:    
            titles.append([visualization_intent])
        else:
            titles.append([""])

    processed_features = torch.tensor(processed_features)

    bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    column_model = SingleColBertWithFeature(bert)
    column_model.load_state_dict(torch.load(column_model_path))
    column_model.eval()
    title_single_col = tokenizer(titles, headers, is_split_into_words=True, pad_to_max_length = True, max_length=30, return_tensors="pt")
    output = column_model(title_single_col, processed_features)

    #### 結果出力

    output = output.tolist()[0][:len(data)]
    data = [(i-min(output)) / (max(output) - min(output)) for i in output]

    print("predict visualization type : {} chart".format(Types[predict[0].item()]))
    print("predict visualizatoin columns percent")
    print("header  :  percent")
    for i, percent in enumerate(data):
        print("{}  :  {}".format(headers[i][0], percent))


if __name__=="__main__":
    main()