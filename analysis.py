# -*- coding: utf-8 -*-
"""
Author: Siddharth Gosalia
Date: 4/20/2018
Title: Sentiment Analysis of Amazon Product Reviews

"""
############################################################################
#
#                 DATA LOADING AND PRE-PROCESSING
#
############################################################################
import pandas as pd

i = 0
reviewText = ""
line_output = []

with open("C:\\Users\\gosal\\Desktop\\IST 664\\clothing_shoes_jewelry.txt") as f:
    for line in f:
        if line.startswith("reviewText"):
            lines = line.split("reviewText:")[1]
            #print(lines)
            line_output.append(lines)
            i = i + 1
        if i >= 8000:
            break

series = pd.Series(line_output)
df = pd.DataFrame()
df['reviewText'] = series.values

#df.to_csv("reviews.csv", sep='\n', header = False, index = False)


############################################################################
#
#                 FEATURE EXTRACTION ON TRAINING DATA
#
############################################################################

import nltk
#nltk.download('sentence_polarity')
from nltk.corpus import sentence_polarity
import random

sentences = sentence_polarity.sents()
print(len(sentences))
print(type(sentences))
print(sentence_polarity.categories())

## setup the movie reviews sentences for classification
# create a list of documents, each document is list of words in sentence paired with category
documents = [(sent, cat) for cat in sentence_polarity.categories() 
	for sent in sentence_polarity.sents(categories=cat)]

# get all words from all movie_reviews and put into a frequency distribution
#   note lowercase, but no stemming or stopwords
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)

# get the 2000 most frequently appearing keywords in the corpus
word_items = all_words.most_common(2000)
word_features = [word for (word,count) in word_items]

# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# get features sets for a document, including keyword features and category feature
featuresets = [(document_features(d, word_features), c) for (d, c) in documents]


classifier = nltk.NaiveBayesClassifier.train(featuresets)

############################################################################
#
#            FEATURE EXTRACTION AND SENTIMENT ANALYSIS ON TEST DATA
#
############################################################################
df["tokens"] = df["reviewText"].apply(nltk.word_tokenize)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

j = 0
for row in df["tokens"]:
    df.iloc[j,1] = [w for w in row if not w in stop_words]
    j = j + 1
    
reviewFeatureSets = [(document_features(d, word_features)) for d in line_output]

processedReviews = pd.DataFrame(df["tokens"])

predictions = []
# evaluate the accuracy of the classifier
for i, each in enumerate(reviewFeatureSets):
    temp = classifier.classify(each)
    predictions.append((processedReviews.iloc[i],temp))
    
output = pd.DataFrame(predictions)
output.columns = ['Tokens', 'Sentiment']

reviewSentiment = pd.concat([df["reviewText"], output["Sentiment"]], axis=1)

# Positive Reviews
positiveReviews = reviewSentiment[reviewSentiment["Sentiment"]=='pos']

# Negative Reviews
negativeReviews = reviewSentiment[reviewSentiment["Sentiment"]=='neg']

# Split data into train and test (90:10) and calculate acccurary
train_set, test_set = predictions[7200:], predictions[:800]
print(nltk.classify.accuracy(classifier, test_set))
# 99.375%

fold1, fold2, fold3, fold4, fold5 = predictions[0:1600], predictions[1600:3200], predictions[3200:4800], predictions[4800:6400], predictions[6400:8000]

# 1st Fold as Test and rest as train
train_set_cv1, test_set_cv1 = predictions[1600:8000], predictions[0:1600] 
print(nltk.classify.accuracy(classifier, test_set_cv1))
# 99.25%

# 2nd Fold as Test and rest as train
train_set_cv2, test_set_cv2 = predictions[0:1600 & 3200:8000], predictions[1600:3200]
print(nltk.classify.accuracy(classifier, test_set_cv2))
# 99.312%

# 3rd Fold as Test and rest as train
train_set_cv3, test_set_cv3 = predictions[0:3200 & 4800:8000], predictions[3200:4800]
print(nltk.classify.accuracy(classifier, test_set_cv3))
# 99.06%

# 4th Fold as Test and rest as train
train_set_cv4, test_set_cv4 = predictions[0:4800 & 6400:8000], predictions[4800:6400]
print(nltk.classify.accuracy(classifier, test_set_cv4))
# 99.37%

# 5th Fold as Test and rest as train
train_set_cv5, test_set_cv5 = predictions[0:6400], predictions[6400:8000]
print(nltk.classify.accuracy(classifier, test_set_cv5))
# 98.87%

####################################################################################
# 
#    BONUS: PREDICTING SENTIMENTS USING SCIKIT LEARN LOGISTIC REGRESSION CLASSIFIER
#
####################################################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

# Data extraction
data = []
data_labels = []
for review in positiveReviews["reviewText"] : 
    data.append(review) 
    data_labels.append('pos')
 
for review in negativeReviews["reviewText"] : 
    data.append(review) 
    data_labels.append('neg')

# Feature extraction
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray()

# Split extracted data into train and test data
X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80, 
        random_state=1254)

# Define Logistic Legression Model
log_model = LogisticRegression()

# Fit model to training data
log_model = log_model.fit(X=X_train, y=y_train)

# Predict sentiments on test data
y_pred  = log_model.predict(X_test)

# Model Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
# Accuracy: 99.68%

# Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
# TP: 10
# TN: 1585
# FP: 4
# FN: 1
