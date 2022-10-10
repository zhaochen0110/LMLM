import yake
import pandas as pd
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
# def get_MI_word(file_path, save_path, topk_freq):

def get_frequency_sentence(file_path, save_path, pivot_path, pivot_num, topk):
    with open(pivot_path) as f:
        pivot=[word.strip() for word in f.readlines()][:pivot_num]
    sentence_frequency_index = {}
    if file_path.split('.')[-1] == 'csv':
        df = pd.read_csv(file_path)
        text = ''
        s1 = list(df['sentence1'])
        s2 = list(df['sentence2'])
        label = list(df['label'])
        index = 0
        for i in range(len(s1)):
            num = 0
            text = s1[i] + s2[i]
            text_str = text.split()[:128]
            for word in text_str:
                word = re.sub(u"([^\u0041-\u005a\u0061-\u007a])", "", word)
                if word in pivot:
                    num += 1
            sentence_frequency_index[str(index)] = num
            index += 1
    frequency_index = sorted(sentence_frequency_index.keys(), key=lambda x: sentence_frequency_index[x], reverse=True)[:topk]
    frequency_dict = sorted(sentence_frequency_index.items(), key=lambda x: x[1], reverse=True)[:topk]
    print(frequency_dict)
    sentence1_lst = []
    sentence2_lst = []
    label_lst = []
    for index in frequency_index:
        sentence1_lst.append(s1[int(index)])
        sentence2_lst.append(s2[int(index)])
        label_lst.append(label[int(index)])
    data = {"sentence1": sentence1_lst, "sentence2": sentence2_lst, "label": label_lst}

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

def random_selection(file_path, save_path, num):
    df = pd.read_csv(file_path)
    s1 = list(df['sentence1'])
    s2 = list(df['sentence2'])
    label = list(df['label'])
    index_lst = random.sample(range(0, len(s1)), num)
    sentence1_lst = []
    sentence2_lst = []
    label_lst = []
    for index in index_lst:
        sentence1_lst.append(s1[int(index)])
        sentence2_lst.append(s2[int(index)])
        label_lst.append(label[int(index)])
    data = {"sentence1": sentence1_lst, "sentence2": sentence2_lst, "label": label_lst}
    # 数据初始化成为DataFrame对象
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)





def index2sentence(sentence_path, file_path, save_path):
    with open(sentence_path) as f:
        index =[word.strip() for word in f.readlines()]

    df = pd.read_csv(file_path)
    s1 = list(df['sentence1'])
    s2 = list(df['sentence2'])
    label = list(df['label'])
    sentence1_lst = []
    sentence2_lst = []
    label_lst = []
    for index in index:
        sentence1_lst.append(s1[int(index)])
        sentence2_lst.append(s2[int(index)])
        label_lst.append(label[int(index)])
    data = {"sentence1": sentence1_lst, "sentence2": sentence2_lst, "label": label_lst}
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)





