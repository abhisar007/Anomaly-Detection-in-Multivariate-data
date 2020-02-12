import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime as dt
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from pylab import rcParams
from sklearn.cluster import DBSCAN
import math as math
from sklearn.neighbors import NearestNeighbors
from datetime import datetime



a= pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\AnomalyPaper\data\MSS_DE_VMANAGE\extracted\PcbTrace.csv',error_bad_lines=False, sep=';',header = 0)

df=pd.DataFrame(data=a,columns=['PcbID','TimeDone','McID','CountPCB','DeviceID','Program','CycleTime','NumComp','NumBlocks','NumErrors','OrderNo','Operation','Lane','SerializedID','VariantID','InsertDate','ItemProcessData','Id'])


#print(df.Program.value_counts()) 
 
b=df[df.Program == '9058929-00_T11-3']
b=b.drop(columns=['ItemProcessData','Id','PcbID','Program','SerializedID','VariantID','InsertDate','OrderNo','McID','Lane','DeviceID'])

c=b.dropna()

#label encoding

labelencoder=LabelEncoder()
c['Operation'] = labelencoder.fit_transform(c['Operation'])

c['TimeDone']= pd.to_datetime(c['TimeDone'])

print(c.dtypes)

#to select by year
#data = c[c['TimeDone'].dt.year == 2017]


data=c.sort_values(by='TimeDone')



testingSeries = data.set_index('TimeDone')

testingSeries.plot()
plt.show()


cycle_time=pd.DataFrame(data=testingSeries,columns=['CycleTime']) 


cols=['CountPCB','CycleTime','NumComp','NumBlocks','NumErrors','Operation']



for i in cols:
    plt.scatter(testingSeries.index,testingSeries[i], alpha=0.5)
    plt.title('Scatter plot')
    plt.xlabel('TimeDone')
    plt.ylabel(i)
    plt.show()
    
    
#function to decomepose time series to analyze trends, seasonality and residue
def trendAnalysis(data):
    rcParams['figure.figsize'] = 11, 9
    decomposition = sm.tsa.seasonal_decompose(data,freq = 4, model='additive') 
    decomposition.plot()  
    plt.title('Quaterly trend and seasonality for Cycle Time' )
    plt.show()

trendAnalysis(cycle_time)

import seaborn as sns

#Using Pearson Correlation
def corr(data):
    plt.figure(figsize=(10,8))
    #testingSeries
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    cor_target = abs(cor["CycleTime"])
    
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.5]
    print(relevant_features)
    
    #print(data[["NumComp","NumBlocks"]].corr())
    
    
 


    
unixtimestamp=[]

for i in data['TimeDone']:
    timestamp = datetime.timestamp(i)
    timestamp=int(timestamp)
    unixtimestamp.append(timestamp)
    
unixtestingseries= testingSeries 
unixtestingseries['unixtime']= unixtimestamp
testingSeries=testingSeries.drop(columns=['unixtime'])
    
corr(testingSeries)
corr(unixtestingseries)   

from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

def histQQplot(data):
    
    for i in cols:
        print(i)
        #pyplot.hist(testingSeries[i])
        pyplot.hist(data[i])
        pyplot.show()
    

        #qqplot(testingSeries[i], line='s')
        qqplot(data[i], line='s')
        pyplot.show()


#log_data = np.log(testingSeries)    
#
#pyplot.hist(testingSeries['CycleTime'])
#pyplot.show()
#pyplot.hist(testingSeries['CountPCB'])
#pyplot.show()


cycletime5000=testingSeries['CycleTime'].iloc[9998:]
testingSeries5000=testingSeries.iloc[9998:]

histQQplot(testingSeries5000)


# Shapiro-Wilk Test

from scipy.stats import shapiro


#p<= alpha: reject H0, not normal.
#p> alpha: fail to reject H0, normal.


# normality test
stat, p = shapiro(cycletime5000)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
    
    
#chi-square test for normality in discrete data

from scipy.stats import chisquare
stats, pval =chisquare(cycletime5000)

print(stats)
print(pval)

