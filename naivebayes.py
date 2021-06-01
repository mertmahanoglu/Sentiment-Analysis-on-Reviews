# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:10:19 2021

@author: Mert
"""

import pandas as pd
import csv


df = pd.read_csv('yeniveriseti.csv', encoding = 'utf-8-sig' )

df = df[['comments', 'sentiments']]

df.columns = ['comments', 'sentiments']

        
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
vectorizer = CountVectorizer(max_features=2000)


X = df.comments.values
y = df.sentiments.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


vectorizer.fit(X_train.astype("str"))

X_train = vectorizer.transform(X_train.astype("str"))
X_test = vectorizer.transform(X_test.astype("str"))

from sklearn.naive_bayes import BernoulliNB

mnb = BernoulliNB()
mnb.fit(X_train.toarray(),y_train)

y_pred = mnb.predict(X_test.toarray())


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report
print('\n Precision: {}%'.format(precision_score(y_test, y_pred)))
print('\n Recall: {}%'.format(recall_score(y_test, y_pred)))
print('\n Accuracy: {}%'.format(accuracy_score(y_test, y_pred)))
print('\n F1 Score: {}%'.format(f1_score(y_test, y_pred)))

print('\n Train Acc: {}%'.format(accuracy_score(y_train,  mnb.predict(X_train.toarray()))))
print('\n Test Acc: {}%'.format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test,y_pred))

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      