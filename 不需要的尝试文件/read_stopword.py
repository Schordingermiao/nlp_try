import os
#print(os.getcwd())
#print(os.listdir())

with open ('./stopwords-master\stopwords-master/cn_stopwords.txt', encoding='UTF-8') as f:
    stopw=f.read().split('\n')
    f.close()
print(stopw)