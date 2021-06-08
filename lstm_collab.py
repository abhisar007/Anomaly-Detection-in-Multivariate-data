from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(1)

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from pylab import rcParams
from sklearn.cluster import DBSCAN
import math as math
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import numpy.matlib
from sklearn import preprocessing
import seaborn as sns
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import chisquare
from scipy.stats import shapiro 
import time 
from sklearn.preprocessing import LabelEncoder
plt.style.use('ggplot')

start_time = time.time()

a= pd.read_csv(r'/content/drive/My Drive/newDataFileMC_11.csv', error_bad_lines=False, sep=',', header = 0)

df=pd.DataFrame(data=a, columns=['PcbID','TimeDone','McID','CountPCB','DeviceID','Program','CycleTime','NumComp','NumBlocks','NumErrors','OrderNo','Operation','Lane','SerializedID','VariantID','InsertDate','ItemProcessDataId'])

#print(df.Program.value_counts())

df=df.drop(columns=['ItemProcessDataId','PcbID','Program','SerializedID','VariantID','InsertDate','OrderNo','McID','SerializedID','Lane','DeviceID'])

c=df.dropna()
#label encoding
c=c.drop_duplicates()

labelencoder=LabelEncoder()
c['Operation'] = labelencoder.fit_transform(c['Operation'])

c['TimeDone']= pd.to_datetime(c['TimeDone'])

data=c.sort_values(by='TimeDone')
testingSeries = data.set_index('TimeDone')



#..............................................................................    
unixtimestamp=[]

for i in data['TimeDone']:
    timestamp = datetime.timestamp(i)
    timestamp=int(timestamp)
    unixtimestamp.append(timestamp)
    
unixtestingseries= testingSeries 
unixtestingseries['unixtime']= unixtimestamp
testingSeries=testingSeries.drop(columns=['unixtime'])
    
# getting length of list 
length = len(unixtimestamp)
tackTime=[0] 
 
for io in range (1, len(unixtimestamp)):
    tdiff=unixtimestamp[io]- unixtimestamp[io-1]
    tackTime.append(tdiff)
    

plt.plot(testingSeries.index, tackTime, color='g')
plt.xlabel('TimeDone')
plt.ylabel('TackTime')
plt.title('TackTime over TimeDone for Program: 9058929-00_T11-3 ')
plt.show()

testingSeries["TackTime"]=np.array(tackTime)
#..............................................................................

def min_max_scaler(data):   
    mm_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #data= mm_scaler.fit_transform(data)
    data = pd.DataFrame(mm_scaler.fit_transform(data), index=data.index, columns=data.columns)
    return data

from sklearn.preprocessing import StandardScaler

def standard_scalar(data):
    scaler = StandardScaler()  
    data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    return data



normalized_testingSeries= standard_scalar(testingSeries)

#normalized_testingSeries=normalized_testingSeries.values
#print(normalized_testingSeries)

#for testing......................................................
train_size = int(len(normalized_testingSeries) * 0.75)
test_size = len(normalized_testingSeries) - train_size
train= normalized_testingSeries.iloc[0:train_size]
test=normalized_testingSeries.iloc[train_size:len(normalized_testingSeries)]

def create_dataset(X, time_steps=1):
    
    Xs = []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
 
    
    return np.array(Xs)
        

TIME_STEPS = 30

# reshape to [samples, time_steps, n_features]

X_train= create_dataset(train,TIME_STEPS)
X_test= create_dataset(test,TIME_STEPS)

print(X_train.shape)



#Autoencoders implementation

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(LSTM(units=32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(LSTM(units=8, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(n=X_train.shape[1]))

#ENCODER ENDS AND DECODER STARTS

model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(TimeDistributed(Dense(units=X_train.shape[2])))

model.compile(loss='mae', optimizer='adam')
model.summary()

history = model.fit(X_train, X_train,epochs=100,batch_size=32,validation_split=0.1,shuffle=False)

X_train_pred = model.predict(X_train)


train_mae_loss = np.mean(np.abs(X_train_pred, X_train), axis=1)

plt.plot(train_mae_loss)
plt.show()


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.close()

accumulated_train_loss=np.sum(train_mae_loss,axis=1).tolist()
sns.distplot(accumulated_train_loss,bins=50,kde=True)



#thresholding

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)


accumulated_test_loss=np.sum(test_mae_loss,axis=1).tolist()
sns.distplot(accumulated_test_loss,bins=50,kde=True)


THRESHOLD = 2.4
test_score=pd.DataFrame(index=test[TIME_STEPS:].index)
#train_score = pd.DataFrame(index=normalized_testingSeries.index)
test_score['loss'] = accumulated_test_loss
test_score['threshold'] = THRESHOLD
test_score['anomaly'] = test_score.loss > test_score.threshold
anomalies = test_score [test_score.anomaly == True]


plt.plot(test_score.index,test_score.loss, label='loss')
plt.plot(test_score.index,test_score.threshold,label='thresholding score')
plt.xticks(rotation=25)
plt.legend()
plt.show()
plt.close()

#inverse transformation is needed before next steps......................................

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
# Define the date format
date_form = DateFormatter("%d-%b-%y")

fig, ax = plt.subplots(figsize=(20, 5))

ax.xaxis.set_major_formatter(date_form)
# Ensure a major tick for each week using (interval=1) 
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))



plt.plot(test[TIME_STEPS:].index,test[TIME_STEPS:].TackTime,label='Tackt Time',linestyle='-', marker='X')
sns.scatterplot(anomalies.index,anomalies.loss,color=sns.color_palette()[3],s=100,label='anomaly')
plt.xticks(rotation=25)
plt.legend()
plt.show()



fig, ax = plt.subplots(figsize=(20, 5))
ax.xaxis.set_major_formatter(date_form)
# Ensure a major tick for each week using (interval=1) 
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
 
plt.plot(test[TIME_STEPS:].index, test[TIME_STEPS:].CycleTime,label='Cycle Time')
sns.scatterplot(anomalies.index,anomalies.loss,color=sns.color_palette()[3],s=100,label='anomaly')
plt.xticks(rotation=25)
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(20, 5))
ax.xaxis.set_major_formatter(date_form)
# Ensure a major tick for each week using (interval=1) 
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.plot(test[TIME_STEPS:].index, test[TIME_STEPS:].NumErrors,label='Number of Errors')
sns.scatterplot(anomalies.index,anomalies.loss,color=sns.color_palette()[3],s=100,label='anomaly')
plt.xticks(rotation=25)
plt.legend()
plt.show()


updated_testScore_df=pd.concat([test, test_score['anomaly']], axis=1)
updated_testScore_df=updated_testScore_df.dropna()
