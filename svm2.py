# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:33:05 2021

@author: Mert
"""

import pandas as pd

df = pd.read_csv('yeniveriseti.csv', encoding = 'utf-8-sig' )

df = df[['comments', 'sentiments']]

df.columns = ['comments', 'sentiments']

X = df.comments.values
y = df.sentiments.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.3,   random_state = 0)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000)

vectorizer.fit(X_train.astype("str"))

X_train = vectorizer.transform(X_train.astype("str"))
X_test = vectorizer.transform(X_test.astype("str"))


from sklearn.svm import SVC

svc = SVC(kernel="sigmoid")

svc.fit(X_train,y_train)

result = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,result)
print(cm)

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report

accuracy = accuracy_score(y_test,result)
print('\n Precision: {}%'.format(precision_score(y_test, result)))
print('\n Recall: {}%'.format(recall_score(y_test, result)))
print('\n Accuracy: {}%'.format(accuracy_score(y_test, result)))
print('\n F1 Score: {}%'.format(f1_score(y_test, result)))


print('\n Train Acc: {}%'.format(accuracy_score(y_train,  svc.predict(X_train))))
print('\n Test Acc: {}%'.format(accuracy_score(y_test, result)))

print(classification_report(y_test,result))





























