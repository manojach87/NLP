import textmining
import nltk
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, STOPWORDS
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
xDIR = 'C:/Users/V574361/Documents/bigdata/files'

#def tdm():
# Create some very short sample documents
# Initialize class to create term-document matrix
tdm = textmining.TermDocumentMatrix()
# Add the documents
for f in os.listdir(xDIR):
    #print(f)
    tdm.add_doc(open(os.path.join(xDIR,f), encoding="utf8").read())
#for row in tdm.rows(cutoff=1):
#    print(row)
#print(tdm)
#    return tdmm
#x = tdm()

tdm.write_csv(filename="C:/Users/V574361/Documents/bigdata/tdm.test.csv")


import pandas
import pandas as pd
cereal_df = pd.read_csv("C:/Users/V574361/Documents/bigdata/tdm.test.csv")
tdm=cereal_df.T
pd.write_csv(tdm,"C:/Users/V574361/Documents/bigdata/tdm.test1.csv")
tdm.to_csv("C:/Users/V574361/Documents/bigdata/tdm.test1.csv", sep=',', encoding='utf-8')

