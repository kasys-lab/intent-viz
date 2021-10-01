import os
import pickle
import spacy
from collections import defaultdict
from tqdm import tqdm
from train import get_vec_dict, get_processed_feature
from sklearn.preprocessing import LabelEncoder

nlp = spacy.load('en_core_web_sm')



def main(): 

    vec_dict = get_vec_dict("/home/maruta14/graduation/tableau_api/glove/glove.6B.100d.txt")


    TITLE_DIR = "prosessed_data/title/honban2/"
    DATA_DIR = 'prosessed_data/use_column_name/honban2/'
    LABEL_DIR = "prosessed_data/mark/honban2/"
    FID_DIR = "prosessed_data/fid/honban2/"
    ALL_DIR = "prosessed_data/all_data_feature/honban2/"
    dataset = defaultdict(list)
    field_dirpath = {
            "title": TITLE_DIR,
            "data": DATA_DIR,
            "label": LABEL_DIR,
            "fid": FID_DIR
            "all_data": ALL_DIR
        }
    print("load data")

    #[header, use_flag, [features]]の形に変換
    for field, dirpath in field_dirpath.items():
        filenames = os.listdir(dirpath)        
        for fname in tqdm(sorted(filenames)):
            # print(fname)
            with open(dirpath+fname, "rb") as f:
                content = pickle.load(f)
                if field == "data":
                    for con in content:
                        one_data = []
                        for header, feature in con.items():
                            features = [None if type(fea[1]) == str else fea[1] for fea in feature[1]]
                            features[0:0] = [header, feature[0]]
                            one_data.append(features)
                        dataset[field].append(one_data)
                else:
                    dataset[field].extend(content)


    remove_idx_list = set()
    ### ヒートマップなど削除 ###  
    labels = []
    for idx, label in tqdm(enumerate(dataset["label"])):
        if label == "Heatmap" or label == "Polygon" or label == "Text" or label == "GanttBar":
            remove_idx_list.add(idx)


    dataset["title"] = [title for idx, title in enumerate(dataset["title"]) if not idx in remove_idx_list]
    dataset["label"] = [label for idx, label in enumerate(dataset["label"]) if not idx in remove_idx_list]
    dataset["fid"] = [fid for idx, fid in enumerate(dataset["fid"]) if not idx in remove_idx_list]
    dataset["data"] = [data for idx, data in enumerate(dataset["data"]) if not idx in remove_idx_list]
    dataset["all_data"] = [data for idx, data in enumerate(dataset["all_data"]) if not idx in remove_idx_list]



    nlp = spacy.load('en_core_web_sm')
    processed_titles = []
    duplicate = set()
    remove_idx_list = set()
    remove_list = {".", ",", "(", ")", "[", "]", "{", "}", "=", "!", "?", "*", ' ', '\t', ";", ":", "  "}

    ###タイトルprocess###
    print("title process")
    for idx, title in tqdm(enumerate(dataset["title"])):
        title = nlp(title)
        words = [word.lower_ for word in title if not word.text in remove_list]
        vecs = [vec_dict[word] for word in words if word in vec_dict]
        processed_title = []
        for word in words:
            if word in vec_dict:
                processed_title.append(word)
        join_title = " ".join(processed_title)
        if join_title in duplicate:
            remove_idx_list.add(idx)
            continue

        if len(vecs) < 4 or len(words)/2 > len(vecs):
            remove_idx_list.add(idx)

        else:
            processed_titles.append(processed_title)
            duplicate.add(join_title)
    


    dataset["label"] = [label for idx, label in enumerate(dataset["label"]) if not idx in remove_idx_list]
    dataset["fid"] = [fid for idx, fid in enumerate(dataset["fid"]) if not idx in remove_idx_list]
    dataset["data"] = [data for idx, data in enumerate(dataset["data"]) if not idx in remove_idx_list]
    dataset["all_data"] = [data for idx, data in enumerate(dataset["all_data"]) if not idx in remove_idx_list]

        
    column_count = []
    features = []
    header = []
    header_vec = []
    processed_feature = []
    use_or_not = []

    print("column process", len(dataset["data"]))
    ### 列process ###
    ## 分解
    for idx, data in tqdm(enumerate(dataset["data"])):
        for column in data:
            column_count.append(idx)
            features.append(column[2:])
            use_or_not.append(column[1])
            header_words = nlp(column[0])
            header.append([word.lower_ for word in header_words if not word.text in remove_list])
    ## 各列スケーリング
    print("scaling....")
    processed_feature = get_processed_feature(features)

    ## 合体
    count = 0
    processed_column_features_header = []
    columns = []
    for i, f, t, u in zip(column_count, processed_feature[0], header, use_or_not):
        if count == i:
            processed_column_features_header.append([t, u, f])
        else:
            columns.append(processed_column_features_header)
            processed_column_features_header = []
            count+=1
            processed_column_features_header.append([t, u, f])
    columns.append(processed_column_features_header)

    ## ラベルエンコーディング
    le = LabelEncoder()
    le = le.fit(dataset["label"])
    labels = le.transform(dataset["label"])


    print(len(processed_titles))
    print(len(columns))
    print(len(labels))
    print(len(dataset["fid"]))


    TITLE_PATHNAME = "prosessed_data/title/title_with_use.pickle"
    DATA_PATHNAME = "prosessed_data/use_column_name/column_with_use.pickle"
    LABEL_PATHNAME = "prosessed_data/mark/mark_with_use.pickle"
    FID_PATHNAME = "prosessed_data/fid/fid_with_use.pickle"
    ALL_DATA_PATHNAME = "prosessed_data/all_data_feature/all_data_with_use.pickle"
    path_names = {TITLE_PATHNAME: processed_titles, DATA_PATHNAME: columns, LABEL_PATHNAME: labels, FID_PATHNAME: dataset["fid"]}
    for name, content in path_names.items():
        with open(name, 'wb') as data:
            pickle.dump(content, data)

if __name__== "__main__":
    main()
