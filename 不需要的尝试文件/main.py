
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
text = """我是一条天狗呀！
我把月来吞了，
我把日来吞了，
我把一切的星球来吞了，
我把全宇宙来吞了。
我便是我了！"""
sentences = text.split()
sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
document = [" ".join(sent0) for sent0 in sent_words]
print(document)
# ['我 是 一条 天狗 呀 ！', '我 把 月 来 吞 了 ，', '我 把 日来 吞 了 ，', '我 把 一切 的 星球 来 吞 了 ，', '我 把 全宇宙 来 吞 了 。', '我 便是 我 了 ！']


tfidf_model = TfidfVectorizer().fit(document)
print(tfidf_model.vocabulary_)
# {'一条': 1, '天狗': 4, '日来': 5, '一切': 0, '星球': 6, '全宇宙': 3, '便是': 2}
sparse_result = tfidf_model.transform(document)
print(sparse_result)

tfidf_model2 = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(document)
print(tfidf_model2.vocabulary_)
# {'我': 8, '是': 12, '一条': 1, '天狗': 7, '呀': 6, '把': 9, '月': 13, '来': 14, '吞': 5, '了': 2, '日来': 10, '一切': 0, '的': 15, '星球': 11, '全宇宙': 4, '便是': 3}


# 过滤出现在超过60%的句子中的词语
tfidf_model3 = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.6).fit(document)  
print(tfidf_model3.vocabulary_)
# {'是': 8, '一条': 1, '天狗': 5, '呀': 4, '月': 9, '来': 10, '日来': 6, '一切': 0, '的': 11, '星球': 7, '全宇宙': 3, '便是': 2}


