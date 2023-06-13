import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk

data = pd.read_excel('Dataset.xlsx')
print(data.head())



def remove_tags(string):
    removelist = ""
    try:
        result = re.sub('','',string)          #remove HTML tags
        result = re.sub('https://.*','',result)   #remove URLs
        result=result.replace(".","")
        result = result.lower()
        return result
    except:
        result = str(string).lower()
        return result

data['Comments']=data['Comments'].apply(lambda cw : remove_tags(cw))

import nltk
nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    st = ""
    for w in w_tokenizer.tokenize(text):
        st = st + lemmatizer.lemmatize(w) + " "
    return st
data['Comments'] = data.Comments.apply(lemmatize_text)
#
print(data.head())


s = 0.0
for i in data['Comments']:
    word_list = i.split()
    s = s + len(word_list)
print("Average length of each review : ",s/data.shape[0])
pos = 0
for i in range(data.shape[0]):
    if data.iloc[i]['Annotation'] == 'positive':
        pos = pos + 1
neg = data.shape[0]-pos
print("Percentage of reviews with positive sentiment is "+str(pos/data.shape[0]*100)+"%")
print("Percentage of reviews with negative sentiment is "+str(neg/data.shape[0]*100)+"%")

reviews = data['Comments'].values
labels = data['Annotation'].values
print(labels)
encoder = LabelEncoder()
encoded_labels = encoder.transform(labels)
print(type(encoded_labels))
train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels, stratify = encoded_labels)
# # Hyperparameters of the model
# vocab_size = 3000 # choose based on statistics
# oov_tok = ''
# embedding_dim = 100
# max_length = 200 # choose based on statistics, for example 150 to 200
# padding_type='post'
# trunc_type='post'
# tokenize sentences
# tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
# tokenizer.fit_on_texts(train_sentences)
# word_index = tokenizer.word_index
# convert train dataset to sequence and pad sequences
# train_sequences = tokenizer.texts_to_sequences(train_sentences)
# train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)
# convert Test dataset to sequence and pad sequences
# test_sequences = tokenizer.texts_to_sequences(test_sentences)
# test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)
#
# # model initialization
# model = keras.Sequential([
#     keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#     keras.layers.Bidirectional(keras.layers.LSTM(256)),
#     keras.layers.Bidirectional(keras.layers.LSTM(128)),
#     keras.layers.Bidirectional(keras.layers.LSTM(64)),
#     keras.layers.Bidirectional(keras.layers.LSTM(32)),
#     keras.layers.Bidirectional(keras.layers.LSTM(16)),
#     keras.layers.Dense(24, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
# # compile model
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# # model summary
# model.summary()
#
# m_epochs = 5
# history = model.fit(train_padded, train_labels,
#                     epochs=num_epochs, verbose=1,
#                     validation_split=0.1)
#
# prediction = model.predict(test_padded)
# # Get labels based on probability 1 if p>= 0.5 else 0
# pred_labels = []
# for i in prediction:
#     if i >= 0.5:
#         pred_labels.append(1)
#     else:
#         pred_labels.append(0)
# print("Accuracy of prediction on test set : ", accuracy_score(test_labels,pred_labels))