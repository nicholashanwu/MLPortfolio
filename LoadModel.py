# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:24:17 2020

@author: nwu
"""
import pandas as pd 
import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

sentiment_array = {
    "BHP": 0,
    "CSL": 0,
    "RIO": 0,
    "CBA": 0,
    "WOW": 0,
    "WES": 0,
    "TLS": 0,
    "AMC": 0,
    "BXB": 0,
    "FPH": 0
    }

sentiment_ratio = pd.DataFrame({
    'BHP': pd.Series([0,0,0]),
    'CSL': pd.Series([0,0,0]),
    'RIO': pd.Series([0,0,0]),
    'CBA': pd.Series([0,0,0]),
    'WOW': pd.Series([0,0,0]),
    'WES': pd.Series([0,0,0]),
    'TLS': pd.Series([0,0,0]),
    'AMC': pd.Series([0,0,0]),
    'BXB': pd.Series([0,0,0]),
    'FPH': pd.Series([0,0,0])
})


sentiment_df = []

# load model 
model = tf.keras.models.load_model('saved_model.h5')

# import news 
news = pd.read_json('Trading_news.json')

# retrieves a dict mapping words to their index in the IMDB dataset. 
# Keys are word strings, values are their index
word_index = imdb.get_word_index()

def encode_text(text):
    #converts text to a sequence of words
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]

reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]

def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  
  return (result[0])

def add_sentiment_to_list(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  sentiment_df.append(result[0][0])
  
  
def calculate_sentiment(headline, equity):
    if predict(headline) > 0.5 + standard_deviation:
        sentiment_array[equity] += 1
        
        sentiment_ratio[equity][0] += 1.0
        
    elif predict(headline) < 0.5 - standard_deviation:
        sentiment_array[equity] -= 1
        
        sentiment_ratio[equity][1] += 1.0



for i in range(len(news['Headline'])):
    
    add_sentiment_to_list(news['Headline'][i])

standard_deviation = np.std(sentiment_df)

for i in range(len(news['Headline'])):
    
    calculate_sentiment(news['Headline'][i], news['Equity'][i])
    
for x, y in sentiment_array.items():
    print(x, y)
  
# sentiment_ratio['BHP'] = pd.to_numeric(sentiment_ratio['BHP'], downcast="float")
sentiment_ratio = sentiment_ratio.apply(pd.to_numeric, downcast="float")

for i in range(len(sentiment_array)):
    sentiment_ratio.iat[2, i] = round(float(sentiment_ratio.iat[0, i])/float(sentiment_ratio.iat[1, i]), 3)


sentiment_ratio = sentiment_ratio.rename(index = {0: 'positive', 1: 'negative', 2: 'ratio'})
 # = sentiment_ratio['BHP'][0]/sentiment_ratio['BHP'][1]


print(sentiment_ratio.to_string())    

# text = "good and bad"
# encoded = encode_text(text)
# print(encoded)
# print(predict(text))

# plt.hist(sentiment_df)
# plt.savefig('plot.png')




