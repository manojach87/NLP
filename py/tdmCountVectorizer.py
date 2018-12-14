# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 22:42:11 2018

@author: P2223479
"""
#corpus = synopses
#vectorizer = CountVectorizer()



from sklearn.feature_extraction.text import CountVectorizer

path="C:/Users/Manoj/Documents/bigdata/tdm.vectorizer.csv"

vectorizer=CountVectorizer(max_df=1.0, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 tokenizer=tokenize_and_stem, ngram_range=(1,3))

XV = vectorizer.fit_transform(corpus)
tdmArr=XV.toarray()
numpy.savetxt(path,tdmArr , delimiter=",")

readTDM=pd.read_csv(path, names=vectorizer.get_feature_names())
readTDM.insert(0,'DocName',docs['title'])
readTDM['docname']=docs['title']
readTDM.set_index('docname')
df.rename_axis('docname')
cols = list(range(1, 20))
readTDM=readTDM.drop(readTDM.columns[cols],axis=1)
readTDM.to_csv(path, sep=',', encoding='utf-8')
