from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3

stopwords = nltk.corpus.stopwords.words('english')


xDIR = 'C:/Users/V574361/Documents/bigdata/files'
titles=[]
synopses=[]
i=0
for f in os.listdir(xDIR):
    #print(f)
    if i<3 :
        titles.append(f)
        synopses.append(open(os.path.join(xDIR,f), encoding="utf8").read())
        i=i+1

#titles.append(os.path.join(xDIR,os.listdir(xDIR)[0]))
#titles.append(os.path.join(xDIR,os.listdir(xDIR)[1]))

#synopses.append(open(os.path.join(xDIR,os.listdir(xDIR)[0]), encoding="utf8").read())
#synopses.append(open(os.path.join(xDIR,os.listdir(xDIR)[1]), encoding="utf8").read())

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
            # print(filtered_tokens[len(filtered_tokens)-1])
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


#not super pythonic, no, not at all.
#use extend so it's a big flat list of vocab
def test1(synopses):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in synopses:
        allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
        
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
        
    vframe = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    # print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return [totalvocab_stemmed,totalvocab_tokenized]



def test(synopses):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    allwords_stemmed = tokenize_and_stem(synopses) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
        
    allwords_tokenized = tokenize_only(synopses)
    totalvocab_tokenized.extend(allwords_tokenized)
        
    vframe = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    # print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return vframe
    #[totalvocab_stemmed,totalvocab_tokenized]
    

def checkDF(key,dframe):
    return key in dframe.index

vocab_frame=test(synopses[0])


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

#%time 
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses

# print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans

num_clusters = 3

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

#
joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

docs = { 'title': titles, 'synopsis': synopses, 'cluster': clusters }

frame = pd.DataFrame(docs, index = [clusters] , columns = ['title', 'cluster'])

frame['cluster'].value_counts()

##grouped = frame['title'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

## grouped.mean() #average rank (1 to 100) per cluster



print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
#        print('ind')
#        print(ind)
#        
#        print("terms[ind]")
#        print(terms[ind])
#        print("terms[ind].split(' ')")
#        print(terms[ind].split(' '))
#        print("len(terms[ind])")
#        print(len(terms[ind]))
        wordsToPass=terms[ind].split(' ')
        print(wordsToPass)
        for word in wordsToPass:
            if checkDF(word,vocab_frame)==False:
                wordsToPass.remove(word)
        
        print(wordsToPass)
        
        if len(wordsToPass)>0:
        # vocab_frame.loc[terms[ind].split(' ')]
        
            print(' %s' % 
                  #vocab_frame.loc[terms[ind].split(' ')]
                  vocab_frame.loc[wordsToPass]
                  .values
                  .tolist()[0][0]
                  .encode('utf-8', 'ignore')
                  , end=','
                 )
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.loc[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace
    
print()
print()

