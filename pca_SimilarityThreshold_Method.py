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

a= pd.read_csv(r'C:\baibcf\Desktop\newDataFileMC_11.csv', error_bad_lines=False, sep=',', header = 0)

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


from sklearn.decomposition import PCA
pca_data = PCA(n_components=5)
pca_normalizedData = pca_data.fit_transform(normalized_testingSeries)

print('Explained variation per principal component: {}'.format(pca_data.explained_variance_ratio_))

pca_normalizedData_df=pd.DataFrame(pca_normalizedData,columns=['A','B','C','D','E'])
pca_normalizedData_df=pca_normalizedData_df.values

def cosin_similarity(a, b):
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

picked_Feat_length = len(pca_normalizedData_df)

simListOne=[]
#add first element as zero
simListOne.append(0)

simListTwo=[]

  
for element in range (1, picked_Feat_length):
    
    simListOne.append(cosin_similarity(np.array(pca_normalizedData_df[element]), np.array(pca_normalizedData_df[element-1])))

for row in range (0,(picked_Feat_length-1)):
    
    simListTwo.append(cosin_similarity(np.array(pca_normalizedData_df[row]),np.array(pca_normalizedData_df[row+1])))
    
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
minpts= calcMinPts(pca_normalizedData )
caleps= calcEpsilon(pca_normalizedData ) 
#..............................................................................
#eps: 0.71
db = DBSCAN(eps=caleps , min_samples=minpts).fit(pca_normalizedData ) 

y_pred = db.fit_predict(pca_normalizedData )


plt.scatter(pca_normalizedData [:,0], pca_normalizedData [:,1],s=100, c=y_pred, cmap='Paired')
plt.title("DBSCAN")
plt.show()

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
db_labels = db.labels_

print("cluster information")
print(np.unique(db_labels,return_counts= True))





import hdbscan
#parameters from dbscan may not be relevant
#min_samples= minpts min_samples= minpts, cluster_selection_epsilon= 0.56, cluster_selection_method='leaf', allow_single_cluster=True
clusterer = hdbscan.HDBSCAN(min_samples= minpts,cluster_selection_epsilon= 0.56,cluster_selection_method='leaf', allow_single_cluster=True) 
clusterer.fit(pca_normalizedData)

yp=clusterer.fit_predict(pca_normalizedData)
plt.scatter(pca_normalizedData[:,0], pca_normalizedData[:,1],s=100, c=yp, cmap='Paired')
plt.title("HDBSCAN")
plt.show()

hdscan_labels= clusterer.labels_
print(hdscan_labels.max())

print("cluster information")
print(np.unique(hdscan_labels, return_counts= True))


#..............................................................................
#adding cluster labels  and threshold implementation
#for dbscan
cluster_labels_db= pd.DataFrame(data=db_labels, index=testingSeries.index,columns=['clusterLabel'])     
db_feat_label = pd.concat([testingSeries, cluster_labels_db], axis=1)   

#for hdbscan
cluster_labels_hdb= pd.DataFrame(data=hdscan_labels, index=testingSeries.index,columns=['clusterLabel'])   
hdb_feat_label = pd.concat([testingSeries, cluster_labels_hdb], axis=1) 


#for i in range(db_feat_label['clusterLabel']):
#    cluster_i = db_feat_label((db_feat_label[clusterLabel] == i))

AvgSimList_df=pd.DataFrame(data=AvgSimList, index=testingSeries.index,columns=['similarityScore'])
  
db_feat_label = pd.concat([db_feat_label, AvgSimList_df], axis=1)   
hdb_feat_label = pd.concat([testingSeries, AvgSimList_df], axis=1) 



#db_feat_label['clusterLabel'].iloc

###########################################################################################

#tempdf = db_feat_label[(db_feat_label['clusterLabel']==0)]
#
#sim= tempdf['similarityScore']
#threshold= 0.4 * sim.max()
#tempdf['threshold']=threshold
#tempdf['anomaly']= tempdf.similarityScore < tempdf.threshold
#anomalies = tempdf [tempdf.anomaly == True]


###########################################################################################

clus_ele=np.unique(db_labels)

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

        tempdf = db_feat_label[(db_feat_label['clusterLabel']==i)]
        print(tempdf.shape )
        sim= tempdf['similarityScore']    
        threshold= 0.6 * sim.max()
        
        tempdf['threshold']=threshold
        tempdf['anomaly']= tempdf.similarityScore < tempdf.threshold
        
        anomalies = tempdf [tempdf.anomaly == True]
        
        test_score=test_score.append(tempdf)
        
        anomalies_df=anomalies_df.append(anomalies)



