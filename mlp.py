import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


df = pd.read_csv('yeniveriseti.csv', encoding = 'utf-8-sig' )

df = df[['comments', 'sentiments']]

df.columns = ['comments', 'sentiments']

#strip_Accent  = Ön işleme adımı sırasında accentleri kaldırın ve diğer karakter normalleştirmelerini gerçekleştirin.
#ngram_range= Çıkarılacak farklı kelime n-gramları veya karakter-gramları için n-değerleri aralığının alt ve üst sınırı.
countvec = CountVectorizer(ngram_range=(1,4), 
                           strip_accents='unicode',
                           max_features=2000)


X = df.comments.values
y = df.sentiments.values

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.3,   random_state = 0)

X_train = countvec.fit_transform(X_train.astype("str"))
X_test = countvec.transform(X_test.astype("str"))


#Verbose = İlerleme mesajlarını konsola yazdırıp yazdırılmayacağı
#Hidden_layer_sizes = Bu parametre, Sinir Ağı Sınıflandırıcısında sahip olmak istediğimiz katman sayısını ve düğüm sayısını ayarlamamıza izin verir. 
#Demetteki her öğe, i'nin başlığın indeksi olduğu i'nci konumdaki düğüm sayısını temsil eder. 
#Bu nedenle, demet uzunluğu, ağdaki toplam gizli katman sayısını gösterir.
mlp = MLPClassifier(hidden_layer_sizes=(400,400,100),activation = 'relu', verbose=True)

t = mlp.fit(X_train, y_train)




y_pred = mlp.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

print('\n Precision: {}%'.format(precision_score(y_test, y_pred)))
print('\n Recall: {}%'.format(recall_score(y_test, y_pred)))
print('\n Accuracy: {}%'.format(accuracy_score(y_test, y_pred)))
print('\n F1 Score: {}%'.format(f1_score(y_test, y_pred)))

print('\n Train Acc: {}%'.format(accuracy_score(y_train,  mlp.predict(X_train))))
print('\n Test Acc: {}%'.format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test,y_pred))



                                                  