
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
import pandas as pd

#载入接下来分析用的库
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
'''
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
'''
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
'''
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
'''
from nltk import word_tokenize




from sklearn.cluster import AgglomerativeClustering

#from sklearn.model_selection import train_test_split


# 在标准输出上显示进度日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# 解析命令行参数
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


# Jupyter Notebook和IPython控制台的解决方法
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

Data = pd.read_csv('ChnSentiCorp_htl_all.csv').dropna()
Data=pd.DataFrame(Data)
# #############################################################################
# 从训练集中加载一些类别
#categories = [0,1]
# 下面这行取消注释以使用更大的注释集（超过11k个文档）
# categories = None

print("Loading dataset for categories:")
#print(categories)

dataset =  list(Data["review"])

print("%d documents" % len(dataset))
#print("%d categories" % len(categories))
print()

#labels = categories
#true_k = np.unique(labels).shape[0]
true_k=2#选2个聚类中心


with open ('./stopwords-master/stopwords-master/cn_stopwords.txt', encoding='UTF-8') as f:
    stopw=f.read().split('\n')
    f.close()
#print(stopw)



print("Extracting features from the training dataset "
      "using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # 在HashingVectorizer的输出上执行IDF归一化
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words=stopw, alternate_sign=False,
                                   norm=None)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words=stopw,
                                       alternate_sign=False, norm='l2')
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words=stopw,
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()






# #############################################################################
# 做聚类

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=10,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=10,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()





'''
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()
'''

if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names_out()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()






Data['cluster_label']=km.labels_
#print(Data.head(5))


Data1=Data[["review","cluster_label"]]

#print(Data1.head(5))
'''
X_train, X_test, y_train, y_test = \
    train_test_split(Data1["review"], Data["cluster_label"], test_size=0.33, random_state=42)

train_data=pd.concat([X_train,y_train],axis=1)

test_data=pd.concat([X_test,y_test],axis=1)
'''






import jieba
#jieba.enable_parallel(64) #并行分词开启
Data1['review_cut'] = Data1['review'].apply(lambda i:jieba.cut(i) )

Data1['review_cut'] =[' '.join(i) for i in Data1['review_cut']]

def multiclass_logloss(actual, predicted, eps=1e-15):
    """对数损失度量（Logarithmic Loss  Metric）的多分类版本。
    :param actual: 包含actual target classes的数组
    :param predicted: 分类预测结果矩阵, 每个类别都有一个概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota




xtrain, xvalid, ytrain, yvalid = train_test_split(Data1['review_cut'].values, Data1["cluster_label"], 
                                                  stratify=Data1["cluster_label"], 
                                                  random_state=42, 
                                                  test_size=0.3, shuffle=True)

print (xtrain.shape)
print (xvalid.shape)

#1111111111111111
#非常基础的模型（very first model）基于 TF-IDF (Term Frequency - Inverse Document Frequency)+逻辑斯底回归（Logistic Regression）
def number_normalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
    对于许多实际应用场景来说，以数字开头的tokens不是很有用，
    但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
    
# 利用刚才创建的NumberNormalizingVectorizer类来提取文本特征，注意里面各类参数的含义，自己去sklearn官方网站找教程看


tfv = NumberNormalizingVectorizer(min_df=3,  
                                  max_df=0.5,
                                  max_features=None,                 
                                  ngram_range=(1, 2), 
                                  use_idf=True,
                                  smooth_idf=True,
                                  stop_words = stopw)

# 使用TF-IDF来fit训练集和测试集（半监督学习）
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)

#利用提取的TFIDF特征来fit一个简单的Logistic Regression 
clf = LogisticRegression(C=1.0,solver='lbfgs',multi_class='multinomial')
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
#print(classification_report(predictions, yvalid))


#222222222
#使用词汇计数（Word Counts）作为功能，而不是使用TF-IDF
ctv = CountVectorizer(min_df=3,
                      max_df=0.5,
                      ngram_range=(1,2),
                      stop_words =stopw)

# 使用Count Vectorizer来fit训练集和测试集（半监督学习）
ctv.fit(list(xtrain) + list(xvalid))
xtrain_ctv =  ctv.transform(xtrain) 
xvalid_ctv = ctv.transform(xvalid)

#利用提取的word counts特征来fit一个简单的Logistic Regression 

clf = LogisticRegression(C=1.0,solver='lbfgs',multi_class='multinomial')
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#print(classification_report(predictions, yvalid))



#33333333
#朴素贝叶斯
#利用提取的TFIDF特征来fitNaive Bayes
clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#print(classification_report(predictions, yvalid))


#4444444
#基于词汇计数的基础上使用朴素贝叶斯模型

#利用提取的word counts特征来fitNaive Bayes
clf = MultinomialNB()
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict_proba(xvalid_ctv)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#print(classification_report(predictions, yvalid))






#55555555

#支持向量机（SVM）。

#由于SVM需要花费大量时间，因此在应用SVM之前，我们将使用奇异值分解（Singular Value Decomposition ）来减少TF-IDF中的特征数量。

#使用SVD进行降维，components设为120，对于SVM来说，SVD的components的合适调整区间一般为120~200 
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

#对从SVD获得的数据进行缩放
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

# 调用下SVM模型
clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#print(classification_report(predictions, yvalid))


#6666666
#xgboost


# 基于tf-idf特征，使用xgboost
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), ytrain)
predictions = clf.predict_proba(xvalid_tfv.tocsc())

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

#print(classification_report(predictions, yvalid))


#7777777
# 基于word counts特征，使用xgboost
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_ctv.tocsc(), ytrain)
predictions = clf.predict_proba(xvalid_ctv.tocsc())

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


#print(classification_report(predictions, yvalid))

#888888888
# 基于tf-idf的svd特征，使用xgboost
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_svd, ytrain)
predictions = clf.predict_proba(xvalid_svd)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


#print(classification_report(predictions, yvalid))




#999999
# 再对经过数据标准化(Scaling)的tf-idf-svd特征使用xgboost
clf = xgb.XGBClassifier(nthread=10)
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))


#print(classification_report(predictions, yvalid))
'''
#1010101010101010
#网格搜索（Grid Search）
mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)

#SVD初始化
svd = TruncatedSVD()
    
# Standard Scaler初始化
scl = preprocessing.StandardScaler()

# 再一次使用Logistic Regression
lr_model = LogisticRegression()

# 创建pipeline 
clf = pipeline.Pipeline([('svd', svd),
                         ('scl', scl),
                         ('lr', lr_model)])

param_grid = {'svd__n_components' : [120, 180],
              'lr__C': [0.1, 1.0, 10], 
              'lr__penalty': ['l1', 'l2']}
'''
'''
# 网格搜索模型（Grid Search Model）初始化
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
'''
'''
# 网格搜索模型（Grid Search Model）初始化
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, refit=True, cv=2)

#fit网格搜索模型
model.fit(xtrain_tfv, ytrain)  #为了减少计算量，这里我们仅使用xtrain
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


nb_model = MultinomialNB()

# 创建pipeline 
clf = pipeline.Pipeline([('nb', nb_model)])

# 搜索参数设置
param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# 网格搜索模型（Grid Search Model）初始化
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# fit网格搜索模型
model.fit(xtrain_tfv, ytrain)  # 为了减少计算量，这里我们仅使用xtrain
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
'''