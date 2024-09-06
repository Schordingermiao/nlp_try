#文件说明：找到和用户输入的评论最相近的评论
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

from sklearn.metrics.pairwise import cosine_similarity

#from sklearn.model_selection import train_test_split


def find_similar_data(input_command):

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

    re=input_command
    Data.loc[len(Data.index)]=[-1,re]
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



    print("fit后，所有的词汇如下：")
    print( vectorizer.get_feature_names_out())
    #print("fit后，训练数据的向量化表示为：")
    ZZ=(X.toarray())

    ZZ=pd.DataFrame(ZZ)

    similarity_matrix=cosine_similarity(ZZ)


    #print(similarity_df)
    #对角线设为0，自身与自身排除
    for i in range(similarity_matrix.shape[0]):
        similarity_matrix[i,i]=0


    top_n=10
    similarity_rows=similarity_matrix.argsort(axis=1)[:,-top_n:].tolist()

    print(type(similarity_rows))

    #for i,similarity_rows_indices in enumerate(similarity_rows):
        #print(f"Top {top_n} similar rows for now {i}:{similarity_rows_indices}")
    i=len(similarity_rows)-1
    print(f"Top {top_n} similar rows for now {i}:{similarity_rows[i]}")
    print(type(similarity_rows[i]))
    #similarity_df=pd.DataFrame(similarity_matrix)
    #print(similarity_df[535].sort_values( ascending=False))
    print(Data.iloc[similarity_rows[i],:])
    return Data.iloc[similarity_rows[i],:]


#re=input("请输入评论：")
#find_similar_data(re)

