
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn import preprocessing
from sklearn.decomposition import PCA
bcdata = pd.read_csv('data.csv')

#Count the missing values: it turns out 'Unnamed:32' contains large number of missing values.
bcdata.isnull().sum(axis = 0)

# We need to have better understanding about the data
bcdata.describe()
# ID is just identification of each patient 
# unnamed is meaningless to be included due to large number of missing values.
# Drop unnecessary data: id, diagnosis and unnamed32

#distribution of diagnosis 
bcdata['diagnosis'] = bcdata['diagnosis'].map({'M': 1, 'B': 0})


df = bcdata.drop(['id','Unnamed: 32'],axis = 1)
df2 = bcdata.drop(['id','diagnosis','Unnamed: 32'],axis = 1)
#first five row of the data

print(df.head(n=5))
print(df.shape) # contains 569 rows and 30 columns now
print(df.mean())


sns.countplot(df['diagnosis'], label = 'Count' )
plt.title('Distribution of diagnosis result')
plt.xlabel('Diagnosis result')
plt.ylabel('Frequency')
######################################
#normalization
data_n1 = preprocessing.normalize(df, norm = 'l1')
######################################
#Scaling
from sklearn.preprocessing import StandardScaler as scaling
scaled = scaling().fit(df)
scaled_data = pd.DataFrame(scaled.transform(df))

#Put the index back to data
scaled_data.index = bcdata.index
scaled_data.columns = df.columns
scaled_data['diagnosis'] = bcdata['diagnosis']
print(scaled_data)

# PCA 
st = PCA(n_components = 2)
newdf = st.fit_transform(scaled_data)

diagnosis_result = bcdata['diagnosis']
plt.figure(figsize = (16,11))
plt.subplot(121)
plt.scatter(newdf[:,0],newdf[:,1], c = diagnosis_result, 
            cmap = "coolwarm", edgecolor = "None", alpha=0.35)
#based on the plot above we can two clusters for diagnosis result.
      
###################################### 
#Feature selection#
#split the data into Malignant and Benign
bcdata3 = bcdata.drop(['id','Unnamed: 32'],axis = 1)
Malignantdf = bcdata3[bcdata3['diagnosis'] ==1]
Benigndf = bcdata3[bcdata3['diagnosis'] ==0]


mean=list(bcdata3.columns[1:11]) 
meanvar = bcdata3.loc[:,mean]

plt.rcParams.update({'font.size':5})
plot, ax = plt.subplots(nrows = 5, ncols =2)
ax = ax.flatten()
for idx,ax in enumerate(ax):
    ax.figure
    
    binwidth= (max(df[mean[idx]]) - min(df[mean[idx]]))/40
    bins = np.arange(min(df[mean[idx]]), max(df[mean[idx]]) + binwidth, binwidth)
    ax.hist([Malignantdf[mean[idx]],Benigndf[mean[idx]]], bins=bins, alpha=0.3, normed=True, label=['Malignant','Benign'], color=['blue','orange'])
    ax.set_title(mean[idx])
plt.tight_layout

#Based on the chart from mean variables above we notice that texture_mean, smoothness_mean, Symmetry_mean and fractal_dimension_mean are not
#useful in predicting cancer type due to the distinction of target result. 

#se
se=list(bcdata3.columns[11:21]) 
sevar = bcdata3.loc[:,se]

plt.rcParams.update({'font.size':5})
plot, ax = plt.subplots(nrows = 5, ncols =2)
ax = ax.flatten()
for idx,ax in enumerate(ax):
    ax.figure
    
    binwidth= (max(df[se[idx]]) - min(df[se[idx]]))/40
    bins = np.arange(min(df[se[idx]]), max(df[se[idx]]) + binwidth, binwidth)
    ax.hist([Malignantdf[se[idx]],Benigndf[se[idx]]], bins=bins, alpha=0.3, normed=True, label=['Malignant','Benign'], color=['blue','orange'])
    ax.set_title(se[idx])
plt.tight_layout

#Based on the chart from se variables above we notice that none of the variables from se are not useful in predicting cancer type due to the distinction of target result. 

#worst
worst=list(bcdata3.columns[21:31]) 
worstvar = bcdata3.loc[:,worst]

plt.rcParams.update({'font.size':5})
plot, ax = plt.subplots(nrows = 5, ncols =2)
ax = ax.flatten()
for idx,ax in enumerate(ax):
    ax.figure
    
    binwidth= (max(df[worst[idx]]) - min(df[worst[idx]]))/40
    bins = np.arange(min(df[worst[idx]]), max(df[worst[idx]]) + binwidth, binwidth)
    ax.hist([Malignantdf[worst[idx]],Benigndf[worst[idx]]], bins=bins, alpha=0.6, normed=True, label=['Malignant','Benign'], color=['blue','orange'])
    ax.set_title(worst[idx])
plt.tight_layout

#Based on the chart from worst variables above we notice that all variables most of the variables are not usefull except
#radius_worst, perimeter_worst, area_worst

#Check on multicollinearity 
#Start from mean variables

 # heatmap : https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
for hm in (meanvar,sevar,worstvar):
    sns.set(font_scale=0.8)
    f,ax = plt.subplots(figsize=(10 ,10))   
    sns.heatmap(hm.corr(), annot=True,cmap="YlGnBu" ,cbar = True, linewidths=.3, fmt= '.1f',xticklabels=2, ax=ax, robust = True)


#Plot three heatmaps above to check which variables are potentially highly correlated. 
# As it can be seen from the heatmap plot, radius_mean, perimeter_mean and area_mean are highly correlated with coefficient 1
# Same as se variables and worst variables.


#Drop the highly correlated variables


cl_bcdata3 = bcdata3.drop(['compactness_mean','concave points_mean',
                           'radius_se','perimeter_se','compactness_se','concave points_se',
                           'compactness_worst','perimeter_worst','radius_worst','concave points_worst'
                           ],axis = 1)

# Lets check the correlation again
sns.set(font_scale=0.8)
f,ax = plt.subplots(figsize=(10 ,10))   
sns.heatmap(cl_bcdata3.corr(), annot=True,cmap="YlGnBu" ,cbar = True, linewidths=.1, fmt= '.1f',xticklabels=2, ax=ax, robust = True)

# multicollinearity still exists

cl_bcdata4 = cl_bcdata3.drop(['concavity_worst','texture_worst'],axis = 1)

#again.
sns.set(font_scale=0.8)
f,ax = plt.subplots(figsize=(10 ,10))   
sns.heatmap(cl_bcdata4.corr(), annot=True,cmap="YlGnBu" ,cbar = True, linewidths=.1, fmt= '.1f',xticklabels=2, ax=ax, robust = True)


# Now I think multicollineraity problem has been solved

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt 


from sklearn.preprocessing import StandardScaler as scaling
scaled = scaling().fit(cl_bcdata4)
scaled_data = pd.DataFrame(scaled.transform(cl_bcdata4))


train, test = train_test_split(cl_bcdata4 , test_size = 0.3)

train_X = train.loc[:, train.columns != 'diagnosis']
train_y = train.diagnosis

test_X = test.loc[:, test.columns != 'diagnosis']
test_y = test.diagnosis

# KNN 
one_to_50=list(range(1,50))
accuracy=[]
for k in one_to_50:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn, train_X,train_y,cv=10,scoring='recall')
    accuracy.append(scores.mean())
print(np.round(accuracy,3))
# when K = 5, we got 87.3% accuracy which is good based on the result,but can we optimize it a little bit?
# yes
# Based on the insights we approved from the distinction between Mlignant and Benign, we need to 
# drop more vaiables before we build our model
cl_bcdata5 = cl_bcdata4.drop(['texture_mean','smoothness_mean','symmetry_mean','fractal_dimension_mean',
                              'texture_se','area_se','smoothness_se','concavity_se','symmetry_se','fractal_dimension_se',
                              'smoothness_worst','symmetry_worst','fractal_dimension_worst'],axis=1)

#Drop target for scaling
cl_bcdata6 = cl_bcdata5.drop('diagnosis',axis=1)

from sklearn.preprocessing import StandardScaler as scaling
scaled = scaling().fit(cl_bcdata6)
scaled_data = pd.DataFrame(scaled.transform(cl_bcdata6))

#put index back
scaled_data.index = bcdata.index
scaled_data.columns = cl_bcdata6.columns
scaled_data['diagnosis'] = bcdata['diagnosis']
print(scaled_data)

train, test = train_test_split(scaled_data , test_size = 0.3)

train_X = train.loc[:, train.columns != 'diagnosis']
train_y = train.diagnosis

test_X = test.loc[:, test.columns != 'diagnosis']
test_y = test.diagnosis

one_to_50=list(range(1,50))
accuracy=[]
for k in one_to_50:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn, train_X,train_y,cv=10,scoring='recall')
    accuracy.append(scores.mean())
print(np.round(accuracy,3))


plt.plot(one_to_50, accuracy, color='g')
plt.xlabel('Accruacy of prediction')
plt.ylabel('from 1 to 50')
plt.title('KNN Accruacy')
plt.show()

# RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RM_model = RandomForestClassifier(n_estimators = 100, random_state=43)      
RM_model = RM_model.fit(train_X,train_y)

RM_model.score(test_X, test_y)
print(RM_model.score(test_X, test_y))

cm = confusion_matrix(test_y,RM_model.predict(test_X))
sns.heatmap(cm,annot=True,fmt="d")

#