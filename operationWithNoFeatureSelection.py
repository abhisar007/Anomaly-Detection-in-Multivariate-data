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
from skfeature.function.similarity_based import lap_score,SPEC
from skfeature.function.sparse_learning_based import NDFS
from sklearn import preprocessing
import seaborn as sns
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import chisquare
from scipy.stats import shapiro 
import time 
from sklearn.preprocessing import LabelEncoder

start_time = time.time()

a= pd.read_csv(r'C:\Users\baibcf\Desktop\newDataFileMC_11.csv', error_bad_lines=False, sep=',', header = 0)

df=pd.DataFrame(data=a, columns=['PcbID','TimeDone','McID','CountPCB','DeviceID','Program','CycleTime','NumComp','NumBlocks','NumErrors','OrderNo','Operation','Lane','SerializedID','VariantID','InsertDate','ItemProcessDataId'])

#print(df.Program.value_counts())

df=df.drop(columns=['ItemProcessDataId','PcbID','Program','SerializedID','VariantID','InsertDate','OrderNo','McID','SerializedID','Lane','DeviceID','Operation'])

c=df.dropna()
#label encoding
c=c.drop_duplicates()

#labelencoder=LabelEncoder()
#c['Operation'] = labelencoder.fit_transform(c['Operation'])

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
#normalized_testingSeries=normalized_testingSeries.drop_duplicates()
#Laplacian feature selection..............

#dataLaplacian=normalized_testingSeries.values
#
## k is selected based on the value of !!! minpts !!!!		     
#kwargs_W = {"metric":"euclidean ","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':1}
#
##Construct affinity matrix
#W = construct_W.construct_W(dataLaplacian, **kwargs_W)
#
## generate feature score using affinity matrix		     
#feature_score = lap_score.lap_score(dataLaplacian, W = W)
#print(feature_score)
#
##Rank features based on feature's individual score
#feature_rank = lap_score.feature_ranking(feature_score)
#
#print(feature_rank)
#
## specify number of features to choose		     
#pick_feature = 5
#
## selected features	     
#picked_features = dataLaplacian[:, feature_rank[0:pick_feature]]
#
#simDataFrame=pd.DataFrame(data=picked_features,columns=['NumBlocks','NumComp','TackTime','NumErrors','CycleTime'],index= normalized_testingSeries.index)

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

#noDuplicateSimDataFrame=simDataFrame.drop_duplicates()

noDuplicateSimDataFrame=normalized_testingSeries
datanoDuplicateSimDataFrame=noDuplicateSimDataFrame.values

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#..............................................................................
#Similarity calcualtion between neighbouring instances

def cosin_similarity(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return abs(dot_product / (norm_a * norm_b))

picked_Feat_length = len(datanoDuplicateSimDataFrame)

simListOne=[]
#add first element as zero
simListOne.append(0)

simListTwo=[]

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print(np.array(datanoDuplicateSimDataFrame[0]))
print(np.array(datanoDuplicateSimDataFrame[1]))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
  
for element in range (1, picked_Feat_length):
    
    simListOne.append(cosin_similarity(np.array(datanoDuplicateSimDataFrame[element]), np.array(datanoDuplicateSimDataFrame[element-1])))

for row in range (0,(picked_Feat_length-1)):
    
    simListTwo.append(cosin_similarity(np.array(datanoDuplicateSimDataFrame[row]),np.array( datanoDuplicateSimDataFrame[row+1])))
    
def average(a,b):
    k=np.add(a,b)
    return k/2

#adding the last element as zero
simListTwo.append(0)

AvgSimList=average(simListOne,simListTwo)


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

#..............................................................................
minpts= calcMinPts(datanoDuplicateSimDataFrame)
caleps= calcEpsilon(datanoDuplicateSimDataFrame) 
#..............................................................................
#eps: 0.71
db = DBSCAN(eps=caleps, min_samples=minpts).fit(datanoDuplicateSimDataFrame) 

#y_pred = db.fit_predict(datanoDuplicateSimDataFrame)
#plt.scatter(dataLaplacian[:,0], noDuplicateSimDataFrame[:,1],s=100, c=y_pred, cmap='Paired')
#plt.title("DBSCAN")
#plt.show()

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
db_labels = db.labels_

print("cluster information")
print(np.unique(db_labels,return_counts= True))

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

import hdbscan
#min_samples= minpts min_cluster_size=8
clusterer = hdbscan.HDBSCAN( min_samples= minpts , cluster_selection_epsilon=32.22, cluster_selection_method='leaf', allow_single_cluster=True) 
clusterer.fit(datanoDuplicateSimDataFrame)

#yp=clusterer.fit_predict(noDuplicateSimDataFrame)
#plt.scatter(dataLaplacian[:,0], noDuplicateSimDataFrame[:,1],s=100, c=yp, cmap='Paired')
#plt.title("HDBSCAN")
#plt.show()

#clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#clusterer.condensed_tree_.plot()
#clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

hdscan_labels= clusterer.labels_
print(hdscan_labels.max())

print("cluster information")
print(np.unique(hdscan_labels, return_counts= True))



AvgSimList_df=pd.DataFrame(data=AvgSimList, index=testingSeries.index,columns=['similarityScore'])

#..............................................................................
#adding cluster labels  and threshold implementation
#for dbscan
cluster_labels_db= pd.DataFrame(data=db_labels, index=noDuplicateSimDataFrame.index,columns=['clusterLabel'])     
db_feat_label = pd.concat([noDuplicateSimDataFrame, cluster_labels_db], axis=1)   
db_feat_label = pd.concat([db_feat_label, AvgSimList_df], axis=1)   

#for hdbscan
cluster_labels_hdb= pd.DataFrame(data=hdscan_labels, index=noDuplicateSimDataFrame.index,columns=['clusterLabel'])   
hdb_feat_label = pd.concat([noDuplicateSimDataFrame, cluster_labels_hdb], axis=1) 
hdb_feat_label = pd.concat([hdb_feat_label, AvgSimList_df], axis=1)

###########################################################################################



clus_ele=np.unique(hdscan_labels)

#index=db_feat_label.index
test_score=pd.DataFrame()
anomalies_df=pd.DataFrame()


for i in clus_ele:
    if (i == -1):
        tempdf = hdb_feat_label[(hdb_feat_label['clusterLabel']==i)]
        threshold= tempdf['similarityScore']
        tempdf['threshold']=threshold
        tempdf['anomaly']= True
    
        anomalies = tempdf[tempdf.anomaly == True]
        
        test_score=test_score.append(tempdf)
        anomalies_df=anomalies_df.append(anomalies)
    
    else:
        
        tempdf = hdb_feat_label[(hdb_feat_label['clusterLabel']==i)]
        print(tempdf.shape )
        sim= tempdf['similarityScore']    
        threshold= 0.6 * sim.max()

        tempdf['threshold']=threshold
        tempdf['anomaly']= tempdf.similarityScore < tempdf.threshold
        
        anomalies = tempdf [tempdf.anomaly == True]
        
        test_score=test_score.append(tempdf)
        
        anomalies_df=anomalies_df.append(anomalies)

test_score=test_score.sort_index()
anomalies_df=anomalies_df.sort_index()

print(test_score.anomaly.value_counts())
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#cols=['TackTime','NumErrors','NumComp','CycleTime','CountPCB']
#
#for i in cols:
#    plt.scatter(uniqueVals.index,uniqueVals[i], alpha=0.5)
#    plt.title('Scatter plot')
#    plt.xlabel('TimeDone')
#    plt.ylabel(i)
#plt.show()    
#
#uniqueVals.plot(kind='density',figsize=(8, 7),title="Kernel Density Estimation plot")
#
##mTRIX SCATTER PLOT
#sns.set(style="ticks")
#sns.pairplot(uniqueVals) 
#
#
##TIMESERIES PLOT
#sns.set(style="darkgrid")
## Plot the responses for different events and regions
#sns.lineplot(data=uniqueVals)
#
##Distribution plot options
#sns.set(style="white", palette="muted", color_codes=True)
#rs = np.random.RandomState(10)
#
## Set up the matplotlib figure
#f, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
#sns.despine(left=True)
#
## Plot a simple histogram with binsize determined automatically
#sns.distplot(uniqueVals['CycleTime'], kde=False, color="b", ax=axes[0, 0])
#
## Plot a kernel density estimate and rug plot
#sns.distplot(uniqueVals['CycleTime'], hist=False, rug=True, color="r", ax=axes[0, 1])
#
## Plot a filled kernel density estimate
#sns.distplot(uniqueVals['CycleTime'], hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])
#
## Plot a histogram and kernel density estimate
#sns.distplot(uniqueVals['CycleTime'], color="m", ax=axes[1, 1])
#
#plt.setp(axes, yticks=[])
#plt.tight_layout()


print("--- %s seconds ---" % (time.time() - start_time))
