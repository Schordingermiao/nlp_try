import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
 # CountVectorizer：对语料库中出现的词汇进行词频统计，相当于词袋模型。
# 操作方式：将语料库当中出现的词汇作为特征，将词汇在当前文档中出现的频率（次数）作为特征值。
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('ChnSentiCorp_htl_all.csv')

tem_data=list(data["review"].dropna())

#print(tem_data[0:5])

#  操作词袋模型：

count = CountVectorizer()

# 语料库
docs = tem_data
# bag是一个稀疏的矩阵。因为词袋模型就是一种稀疏的表示。
bag = count.fit_transform(docs)
# 输出单词与编号的映射关系。
print(count.vocabulary_)
# 调用稀疏矩阵的toarray方法，将稀疏矩阵转换为ndarray对象。
#print(bag)
#print(bag.toarray())

# where映射为编号8  there映射为编号5······
# 编号也是bag.toarray转换来的ndarray数组的索引

matrix=bag.toarray()







# 过滤出现在超过60%的句子中的词语
#tfidf_model3 = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.6).fit(tem_data)  
#print(type(tfidf_model3.vocabulary_))
# 处理缺失值
#data = data.dropna()
 
'''
# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data["review"])
'''

from sklearn.decomposition import PCA
 
pca = PCA(n_components=2)
data_pca = pca.fit_transform(matrix)

from sklearn.cluster import KMeans
 
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(data_pca) 

data["predict"]=clusters

data.to_csv("data.csv")