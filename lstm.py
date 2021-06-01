import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

data = pd.read_csv('yeniveriseti.csv')
# Keeping only the neccessary columns
data = data[['comments','sentiments']]

    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['comments'].values.astype("str"))
X = tokenizer.texts_to_sequences(data['comments'].values.astype("str"))
X = pad_sequences(X)
print(X)

#128, her gömme vektörünün kaç boyuta sahip olması gerektiği gibi, özellik boyutu
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


Y = pd.get_dummies(data['sentiments']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Batch size, bir ağırlık güncellemesi gerçekleştirilmeden önce ağda gösterilecek örnek sayısını sınırlar.
batch_size = 32
t = model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2,validation_data=(X_test,Y_test))


score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

y_pred = model.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)
Y_test = np.argmax(Y_test,axis=1)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,classification_report

print('\n Precision: {}%'.format(precision_score(Y_test, y_pred)))
print('\n Recall: {}%'.format(recall_score(Y_test, y_pred)))
print('\n Accuracy: {}%'.format(accuracy_score(Y_test, y_pred)))
print('\n F1 Score: {}%'.format(f1_score(Y_test, y_pred)))


print(classification_report(Y_test,y_pred))



from matplotlib import pyplot as plt
plt.plot(t.history['accuracy'])
plt.plot(t.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()











