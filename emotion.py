from data import *
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense
from keras.regularizers import l2
from sklearn.metrics import accuracy_score

dataset = read_dataset('Data',dtypes.float32)

X_train = dataset.train.images
X_test = dataset.test.images
y_train = dataset.train.labels
y_test = dataset.test.labels



X_test_s = X_test[0:5000]
X_test_s= np.append(X_test_s,X_test[25000:30000],axis=0)
X_test_s=np.append(X_test_s,X_test[48000:53000],axis=0)
X_test_s=np.append(X_test_s,X_test[75000:80000],axis=0)

y_test_s = y_test[0:5000]
y_test_s=np.append(y_test_s,y_test[25000:30000],axis=0)
y_test_s=np.append(y_test_s,y_test[48000:53000],axis=0)
y_test_s=np.append(y_test_s,y_test[75000:80000],axis=0)



from sklearn import preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_test_s,y_test_s,test_size=0.4,random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

def label_tranform(labels):
    values=[]
    for i in range(0,len(labels)):
            values.append(round(labels[i]))

    return np.asarray(values,np.int32)






X_train = np.reshape(X_train, (X_train_transformed.shape[0], X_train_transformed.shape[1], 1))
X_test = np.reshape(X_test, (X_test_transformed.shape[0], X_test_transformed.shape[1], 1))


model = Sequential()
model.add(GRU(4, input_shape=X_train.shape[1:],W_regularizer=l2(0.01),dropout_W=0.4,dropout_U=0.4,U_regularizer=l2(0.4),b_regularizer=l2(0.4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,nb_epoch=100,batch_size=1,verbose=2)

#make prediction
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
print testPredict
print y_test
print label_tranform(testPredict)
accuracy = accuracy_score(y_test,label_tranform(testPredict))
print "Accuuracy GRU", accuracy

model = Sequential()
model.add(LSTM(4, input_shape=X_train.shape[1:],W_regularizer=l2(0.01),dropout_W=0.4,dropout_U=0.4,U_regularizer=l2(0.4),b_regularizer=l2(0.4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,nb_epoch=100,batch_size=1,verbose=2)


#make prediction
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
print testPredict
print y_test
print label_tranform(testPredict)
accuracy = accuracy_score(y_test,label_tranform(testPredict))
print "Accuuracy LSTM", accuracy

model = Sequential()
model.add(SimpleRNN(4, input_shape=X_train.shape[1:],W_regularizer=l2(0.01),dropout_W=0.4,dropout_U=0.4,U_regularizer=l2(0.4),b_regularizer=l2(0.4)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train,y_train,nb_epoch=100,batch_size=1,verbose=2)

#make prediction
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
print testPredict
print y_test
print label_tranform(testPredict)
accuracy = accuracy_score(y_test,label_tranform(testPredict))
print "Accuuracy RNN", accuracy


