import jieba
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

n_components = 5 #文章分成幾個類別

jieba.set_dictionary(r"D:\OneDrive\文件\Analysis\文字分析相關\dict\dict.txt.big.txt")
jieba.load_userdict(r"D:\OneDrive\文件\Analysis\文字分析相關\dict\user_dict.txt")
with open(r"D:\OneDrive\文件\Analysis\文字分析相關\dict\stop_words.txt",encoding="utf-8") as f:
    stops = f.read().split('\n')

df = pd.read_excel(r"C:\Users\user\Desktop\補述.xlsx",sheet_name="工作表1")
note1 = df["補述"].dropna()
note1_cut = []

for i in note1:
    text = ""
    for words in jieba.cut(i, cut_all=1):
        if words not in stops:
            text += " " + words
    note1_cut.append(text)

count = CountVectorizer(max_df=.1, max_features=5000)
X = count.fit_transform(note1_cut)

lda = LatentDirichletAllocation(n_components=n_components, random_state=123, learning_method='batch') #, random_state=123
X_topics = lda.fit_transform(X)

n_top_words = 10
feature_names = count.get_feature_names()
# print(lda.components_.argsort())

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[j] for j in topic.argsort()[:-n_top_words - 1:-1]]))

note_list = []
title_list = ["補述"]
for c in range(n_components):
    title_list.append("type" + str(c+1) + "p")
title_list.append("type")
note_list.append(title_list)

for n, k in enumerate(note1):
    tem_list = []
    tem_list.append(k)
    type = ""
    for c in range(n_components):
        tem_list.append(X_topics[n, c])
        if X_topics[n, c] == max(X_topics[n]):
            if len(type) > 0:
                type += "、"
            type += str(c+1)
    tem_list.append(type)
    note_list.append(tem_list)

note_df = pd.DataFrame(note_list)
note_df.to_excel(r"C:\Users\user\Desktop\test.xlsx", index=0, header=0)

