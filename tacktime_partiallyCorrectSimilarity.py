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
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score
from sklearn import preprocessing
import seaborn as sns
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import chisquare
from scipy.stats import shapiro 
import time

start_time = time.time()

a= pd.read_csv(r'C:\Users\baibcf\Desktop\newDataFileMC_11.csv', error_bad_lines=False, sep=',', header = 0)

df=pd.DataFrame(data=a, columns=['PcbID','TimeDone','McID','CountPCB','DeviceID','Program','CycleTime','NumComp','NumBlocks','NumErrors','OrderNo','Operation','Lane','SerializedID','VariantID','InsertDate','ItemProcessDataId'])

#print(df.Program.value_counts())

df=df.drop(columns=['ItemProcessDataId','PcbID','Program','SerializedID','VariantID','InsertDate','Operation','OrderNo','McID','SerializedID','Lane','DeviceID'])

c=df.dropna()

c['TimeDone']= pd.to_datetime(c['TimeDone'])
#
#to select by year
#data = c[c['TimeDone'].dt.year == 2017]

data=c.sort_values(by='TimeDone')
testingSeries = data.set_index('TimeDone')

#testingSeries.plot()
#plt.show()

cycle_time=pd.DataFrame(data=testingSeries,columns=['CycleTime']) 

cols=['CountPCB','CycleTime','NumComp','NumBlocks','NumErrors']

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

#trendAnalysis(cycle_time)
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
    
#..............................................................................    
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

cycletime5000=testingSeries['CycleTime'].iloc[0:1000]
testingSeries5000=testingSeries.iloc[0:1000]

#histQQplot(testingSeries5000)

# Shapiro-Wilk Test
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
stats, pval =chisquare(cycletime5000)

print(stats)
print(pval)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler  

def min_max_scaler(data):   
    mm_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    #data= mm_scaler.fit_transform(data)
    data = pd.DataFrame(mm_scaler.fit_transform(data), index=data.index, columns=data.columns)
    return data

def standard_scalar(data):
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    return data
    
def max_Ab_Scalar(data):
    scaler = MaxAbsScaler()
    data= pd.DataFrame( scaler.fit_transform(data),index=data.index, columns=data.columns)
    return data
    
def robust_scalar(data):
    scaler = preprocessing.RobustScaler()
    data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    return data

#log(D) is minPts or Mininum sample

def calcMinPts(dataframe):
    count = dataframe.shape[0]
    D = math.log(count)
    minPts = int(D)
    return minPts

def calcEpsilon(dataframe):
    data= dataframe    
    minPts = calcMinPts(data)
    kneighbour = minPts - 1
    nbrs = NearestNeighbors(n_neighbors=minPts, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    

    d = distances[:, kneighbour]
    #i = indices[:, 0]
    sorted_distances = np.sort(d, axis=None)
    df = pd.DataFrame(data=sorted_distances, columns=['values'])
    
    # converts the dataframe values to a list
    values = list(df['values'])

    # get length of the value set
    nPoints = len(values)
    allkdistpoints = np.vstack((range(nPoints), values)).T

    # Access the first and last point and plot a line between them
    largestkdistpoint = allkdistpoints[0]
    kdistlinevector = allkdistpoints[-1] - allkdistpoints[0]
    kdistlinevectorNorm = kdistlinevector / np.sqrt(np.sum(kdistlinevector ** 2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vectorWithlargestkpoint = allkdistpoints - largestkdistpoint

    scalarProduct = np.sum(vectorWithlargestkpoint * np.matlib.repmat(kdistlinevectorNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, kdistlinevectorNorm)
    vecToLine = vectorWithlargestkpoint - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    maxdistance = np.amax(distToLine)
    # knee/elbow is the point with max distance value
    #idxOfBestPoint = np.argmax(distToLine)

    return maxdistance

normalized_testingSeries= standard_scalar(testingSeries)

#normalized_testingSeries=  max_Ab_Scalar(testingSeries)
#normalized_testingSeries= robust_scalar(testingSeries)
#normalized_testingSeries= min_max_scaler(testingSeries)    

#Laplacian feature selection..............

dataLaplacian=normalized_testingSeries.values

# k is selected based on the value of !!! minpts !!!!		     
kwargs_W = {"metric":"euclidean ","neighbor_mode":"knn","weight_mode":"heat_kernel","k":9,'t':1}

#Construct affinity matrix
W = construct_W.construct_W(dataLaplacian, **kwargs_W)

# generate feature score using affinity matrix		     
feature_score = lap_score.lap_score(dataLaplacian, W = W)
print(feature_score)

#Rank features based on feature's individual score
feature_rank = lap_score.feature_ranking(feature_score)

print(feature_rank)

# specify number of features to choose		     
pick_feature = 5

# selected features	     
picked_features = dataLaplacian[:, feature_rank[0:pick_feature]]

simDataFrame=pd.DataFrame(data=picked_features,columns=['NumBlocks','NumComp','TackTime','NumErrors','CycleTime'],index= normalized_testingSeries.index)

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

#..............................................................................
#Similarity calcualtion between neighbouring instances

def cosin_similarity(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

picked_Feat_length = len(picked_features)

simListOne=[]
#add first element as zero
simListOne.append(0)

simListTwo=[]

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print(np.array(picked_features[0]))
print(np.array(picked_features[1]))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
  
for element in range (1, picked_Feat_length):
    
    simListOne.append(cosin_similarity(np.array(picked_features[element]), np.array(picked_features[element-1])))

for row in range (0,(picked_Feat_length-1)):
    
    simListTwo.append(cosin_similarity(np.array(picked_features[row]),np.array( picked_features[row+1])))
    
def average(a,b):
    k=np.add(a,b)
    return k/2

#adding the last element as zero
simListTwo.append(0)

AvgSimList=average(simListOne,simListTwo)

# scaled between (0-1)
mm_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
normalizedAvgSimList=mm_scaler.fit_transform(AvgSimList.reshape(-1, 1))

plt.plot(normalizedAvgSimList)
plt.show()

#..............................................................................
minpts= calcMinPts(picked_features)
caleps= calcEpsilon(picked_features) 
#..............................................................................
#eps: 0.71
db = DBSCAN(eps=caleps , min_samples=minpts).fit(picked_features) 

y_pred = db.fit_predict(picked_features)
plt.scatter(dataLaplacian[:,0], picked_features[:,1],s=100, c=y_pred, cmap='Paired')
plt.title("DBSCAN")
plt.show()

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
db_labels = db.labels_

print("cluster information")
print(np.unique(db_labels,return_counts= True))

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

import hdbscan
#min_samples= minpts
clusterer = hdbscan.HDBSCAN(min_cluster_size=9, cluster_selection_epsilon= 5, cluster_selection_method='leaf',allow_single_cluster=True) 
clusterer.fit(picked_features)

yp=clusterer.fit_predict(picked_features)
plt.scatter(dataLaplacian[:,0], picked_features[:,1],s=100, c=yp, cmap='Paired')
plt.title("HDBSCAN")
plt.show()

#clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#clusterer.condensed_tree_.plot()
#clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())


hdscan_labels= clusterer.labels_
print(hdscan_labels.max())

print("cluster information")
print(np.unique(hdscan_labels,return_counts= True))

#..............................................................................
#Evaluation pipeline......
#for dbscan
cluster_labels_db= pd.DataFrame(data=db_labels, index=normalized_testingSeries.index,columns=['clusterLabel'])     
db_feat_label = pd.concat([simDataFrame, cluster_labels_db], axis=1)   

#for hdbscan
cluster_labels_hdb= pd.DataFrame(data=hdscan_labels, index=normalized_testingSeries.index,columns=['clusterLabel'])   
hdb_feat_label = pd.concat([simDataFrame, cluster_labels_hdb], axis=1) 

#Evaluation pipeline........................................................................      
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

# split into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

def supportVectorMachines(X_train,y_train,X_test,y_test):
    
    #build a classifier and fit the data
    classifier = svm.SVC(kernel='linear', C=1,gamma=0).fit(X_train, y_train.ravel())
    
    #predict labels for test data
    y_pred = classifier.predict(X_test)
    
    confusionMatrix(y_test, y_pred)
    recieverOperatingCurve(X_test,y_test,classifier)
        
def kNearest(X_train,y_train,X_test,y_test):
    classifier = KNeighborsClassifier(n_neighbors = 5, algorithm = 'auto').fit(X_train, y_train.ravel())
    
    #predict labels for test data
    y_pred = classifier.predict(X_test)
    
    confusionMatrix(y_test, y_pred)
    recieverOperatingCurve(X_test,y_test,classifier)
       
def logisticRegression(): 
    classifier = LogisticRegression().fit(X_train,y_train)
    
    #predict labels for test data
    y_pred=classifier.predict(X_test)
    
    confusionMatrix(y_test, y_pred)
    recieverOperatingCurve(X_test,y_test,classifier)
    
    
def confusionMatrix(y_test, y_pred):
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    class_names = [0,1] 

    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))    

def recieverOperatingCurve(X_test,y_test,classifier):
    
    #ROC CURVE
    y_pred_probability = classifier.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_probability)
    auc = metrics.roc_auc_score(y_test, y_pred_probability)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

print("--- %s seconds ---" % (time.time() - start_time))   
