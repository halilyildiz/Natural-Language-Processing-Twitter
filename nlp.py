# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:54:02 2020

@author: halil
"""


import pandas as pd 

#%% import twitter data

data = pd.read_csv("C://Users//halil//Desktop\Code for Github//nlp-twitter//gender-classifier.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis = 1)
data.dropna(axis = 0, inplace = True)
data.gender = [1 if each == "female" else 0 for each in data.gender]

#%% cleaning data
#regular expression RE sample : [^a-zA-Z]

import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description) # Find the letters from a-z and A-Z replace the rest with " "(space)
description = description.lower() # Capitalizing from uppercase to lowercase

#%% stopwords (irrelavent words)
import nltk # naturel language tool kit
nltk.download('punkt') # download punkt folder
nltk.download("stopwords") # download corpus folder
from nltk.corpus import stopwords # than i import it in corpus folder

description = nltk.word_tokenize(description) # split words 

#%% remove irrelavent words

description = [word for word in description if not word in set(stopwords.words("english"))]

#%% lemmatazation sample { loved => love}

import nltk as nlp
nltk.download('wordnet')
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]

description = " ".join(description)

#%% 
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()   # buyuk harftan kucuk harfe cevirme
    description = nltk.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
#%% bag of words

from sklearn.feature_extraction.text import CountVectorizer # create bag of words method
max_features = 5000
count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("en sık kullanılan kelimeler {} kelimeler {}".format(max_features,count_vectorizer.get_feature_names()))

# %%
y = data.iloc[:,0].values   # male or female classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#%% prediction
y_pred = nb.predict(x_test)
print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))


