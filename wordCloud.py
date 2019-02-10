#  -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql import HiveContext
from nltk.tag import pos_tag
import nltk.data
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
import re
from string import digits
from math import *
from collections import Counter
import datetime
from datetime import date, timedelta
from pyspark.sql.types import * 

sc = SparkContext()
sqlc = HiveContext(sc)

lmtzr = WordNetLemmatizer()
ps = PorterStemmer()

###############POS_TAGS, LEMMATIZE AND STEMMING
def _posTag(message):
	finalcount = pd.DataFrame(columns=['word','frequency','resource_name'])
	i=0
	for row in message:
		print(row)
		propernouns=[]
		lemma=[]
		row = row.lower()
		tagged_sent = pos_tag(row.split())
		propernouns = [word for word, pos in tagged_sent if pos == 'NN' or pos == 'PNN' or pos == 'NNS']
		for lemmaProp in propernouns:
			lemma.append(lmtzr.lemmatize(lemmaProp))
		counts=Counter(lemma)
		counts = pd.DataFrame.from_dict(counts, orient='index').reset_index()
		print(counts)
		counts.columns = ['word', 'frequency']
		counts['resource_name'] = message.index.values[i]
		#counts['date']=dataset.index.values[i]
		counts=pd.DataFrame(counts)
		finalcount=finalcount.append(counts)
		i+= 1
	return finalcount

########RESOURCE_NAME AND LABEL WISE SEPERATION
def sentiment(data,label):
	datasets=pd.DataFrame(data)
	datasets=datasets.dropna()
	datasets['resource_name']=datasets['resource_name'].map(lambda x: re.sub(r'^"|"$', '', x))
	datasets=pd.DataFrame(datasets)
	labels=datasets[datasets['sentiment']==label]
	labels=labels[['resource_name','body']]
	print(labels)
	labels=pd.DataFrame(labels)
	labels['body']=labels['body']+' '
	labels=labels.groupby(['resource_name'])['body'].apply(lambda x: x.sum())
	labels = labels.map(lambda x: re.sub(r'([^\s\w]|_)+', ' ', x))
	labels = labels.map(lambda x: re.sub("\d+", ' ', x))
	return labels

#######IMPORT DATA FROM HIVE
dataset=sqlc.sql("SELECT comments from input_table").dropna().toPandas()

#######POSITIVE DATA
posLabel='positive'
positiveData=_posTag(sentiment(dataset,posLabel))
positiveData['label']='positive'
positiveData['label']=positiveData['label'].astype('str')
print(positiveData)

#######NEGATIVE DATA
negLabel='negative'
negativeData=_posTag(sentiment(dataset,negLabel))
negativeData['label']='negative'
negativeData['label']=negativeData['label'].astype('str')
print(negativeData)

########COMBINE POSITIVE AND NEGATIVE
positiveData = positiveData.append(negativeData)
print(positiveData)

########PICK DATE COLUMN
yesterday = date.today() - timedelta(1)
yesterday=yesterday.strftime("%Y-%m-%d")
positiveData['date_id']=yesterday

positiveData=positiveData[positiveData['word'].map(len)>3]
positiveData=positiveData[positiveData['word']!='https']
positiveData=positiveData[positiveData['word']!='http']
positiveData['date_id']=positiveData['date_id'].astype('str')
print(positiveData)

positiveData.index.name=None
positiveData.to_csv('/home/wordCloud/senti_count.csv',index=False)
schema = StructType([StructField('word', StringType()),
                    StructField('frequency', DoubleType()),
					StructField('resource_name', StringType()),
					StructField('label', StringType()),
					StructField('date_id', StringType())
                    ])

positiveData=sqlc.createDataFrame(positiveData,samplingRatio=0.2,schema=schema)

positiveData.write.save(path='/source/wordCloud', source='parquet', mode='overwrite')