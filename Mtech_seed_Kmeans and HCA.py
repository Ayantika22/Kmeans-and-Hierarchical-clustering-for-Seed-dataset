#!/usr/bin/env python
# coding: utf-8

# # Kmeans and Hierarchical Clustering on SEED dataset

# In[1]:


'''Demonstrating seed dataset on various techniques'''
# Importing library
# Adding Preliminary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Importing Dataset
#To demonstrate various clustering algorithms in python, the Iris dataset will be used which has three classes
# in the dependent variable (three type of Iris flowers) and using this dataset clusters will be formed.
seed = pd.read_csv('Seed_data.csv')
seed


# In[3]:


#Preparing Data
#Here we have the target variable ‘Type’. We need to remove the target variable so that this dataset can be used to work in an unsupervised learning environment. The iloc function is used to get the features we require. We also use .values function to get an array of the dataset.
#(Note that we transformed the dataset to an array so that we can plot the graphs of the clusters).

Y = seed['target']          # Split off classifications
X = seed.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values # Split off features


# In[4]:


# Now we will separate the target variable from the original dataset and again convert it to an array by using numpy.

Y = seed['target']
Y = np.array(Y)


# # Seed dataset clustering plot

# In[7]:


# Visualise Classes
# seed dataset has three classes in target

plt.scatter(X[Y == 0, 0], X[Y == 0, 6], s =80, c = 'orange', label = 'Target 0')
plt.scatter(X[Y == 1, 0], X[Y == 1, 6], s =80,  c = 'yellow', label = 'Target 1')
plt.scatter(X[Y == 2, 0], X[Y == 2, 6], s =80,  c = 'green', label = 'Target 2')
plt.title('Seed dataset plot')
plt.legend()


# # Kmeans Clustering for Seed Dataset

# In[8]:


'''Kmeans is a kind of Unsupervised type of Clustering . It basically takes input from Dataset and predicts the clusters 
accordingly'''

# Wine dataset for KMeans

from sklearn.cluster import KMeans


# In[9]:


# Calculating WCSS (within-cluster sums of squares) 

wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# # Elbow plot (Kmeans) for SEED dataset

# In[10]:


# Plot the WCSS

plt.plot(range(1, 11), wcss)
plt.title('The elbow method for Seed dataset')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[11]:


# Running K-Means Model

cluster_Kmeans = KMeans(n_clusters=3)
model_kmeans = cluster_Kmeans.fit(X)
pred_kmeans = model_kmeans.labels_
pred_kmeans


# # Kmeans Clustering plot for Seed dataset
# 

# In[37]:


# Visualizing Output
# In the above output we got value labels: ‘0’, ‘1’  and ‘2’. For a better understanding, we can visualize these clusters.

plt.scatter(X[pred_kmeans == 0, 5], X[pred_kmeans == 0, 0], s = 80, c = 'orange', label = 'Target 0')
plt.scatter(X[pred_kmeans == 1, 0], X[pred_kmeans == 1, 5], s = 80, c = 'yellow', label = 'Target 1')
plt.scatter(X[pred_kmeans == 2, 0], X[pred_kmeans == 2, 5], s = 80, c = 'green', label = 'Target 2')

plt.title('Kmeans Plot for Seed dataset')

plt.legend()


# In[5]:


# KNN accuracy

seed=pd.read_csv('Seed_data.csv')


# In[6]:


X=seed.iloc[:,:-1].values
y=seed.iloc[:,-1].values


# In[7]:


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[33]:


# Calculating Accuracy score, Confusion matrix, Classification report.

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X=seed.iloc[:,:-1].values
y=seed.iloc[:,-1].values

Xtrain, Xtest, y_train, y_test = train_test_split(X, y)
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

knn = neighbors.KNeighborsClassifier(n_neighbors=14)
knn.fit(Xtrain, y_train)
y_pred = knn.predict(Xtest)

print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred))


# In[34]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[21]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Logistic Regression Accuracy

# In[22]:


#Logistic Regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Logistic Regression :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for LR

# In[23]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # K-Nearest Neighbors Accuracy

# In[44]:


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("K Nearest Neighbors :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for KNN

# In[25]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Support Vector Machine Accuracy

# In[29]:


#Support Vector Machine
from sklearn.svm import SVC
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classifier = SVC()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Support Vector Machine:")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for SVM

# In[30]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Gaussian Naive Bayes Accuracy

# In[33]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
classifier = GaussianNB()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Gaussian Naive Bayes :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for GNB

# In[34]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Decision Tree Classifier Accuracy

# In[36]:


#Decision Tree Classifier
from sklearn.model_selection import train_test_split
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

classifier = DT(criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Decision Tree Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for DTC

# In[37]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Random Forest Classifier Accuracy

# In[42]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RF
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = RF(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for RFC

# In[43]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Hierarchical clustering Analysis (HCA) for Seed dataset

# In[18]:


# Import Library for Hierarchical clustering

import matplotlib.pyplot as plt  
from sklearn.cluster import AgglomerativeClustering


# In[19]:


# Plotting of Dendrogram

import scipy.cluster.hierarchy as sch


# # Hierarchical Dendogram plot for Seed dataset

# In[20]:


#Decide the number of clusters by using this dendrogram
Z = sch.linkage(X, method = 'median')
plt.figure(figsize=(20,7))
den = sch.dendrogram(Z)
plt.title('Dendrogram for the clustering of the dataset seed)')
plt.xlabel('Type')
plt.ylabel('Euclidean distance in the space with other variables')


# In[21]:


# Building an Agglomerative Clustering Model

#Initialise Model

cluster_H = AgglomerativeClustering(n_clusters=3)


# In[22]:



# Modelling the data
model_clt = cluster_H.fit(X)
model_clt
pred1 = model_clt.labels_
pred1


# # Hierarchical cluster plot for Glass dataset

# In[23]:


# Plotting the HCA Cluster

plt.scatter(X[pred1 == 0, 0], X[pred1 == 0, 3], s = 80, c = 'orange', label = 'Target 0')
plt.scatter(X[pred1 == 1, 1], X[pred1 == 1, 4], s = 80, c = 'yellow', label = 'Target 1')
plt.scatter(X[pred1 == 2, 1], X[pred1 == 2, 5], s = 80, c = 'green', label = 'Target 2')
plt.title('Hierarchical Plot for Seed dataset')
plt.legend()


# # Hierarchical clustering Accuracy for Seed dataset

# In[35]:



import sklearn.metrics as sm

target = pd.DataFrame(seed.target)
#based on the dendrogram we have two clusetes 
k =3 
#build the model
HClustering = AgglomerativeClustering(n_clusters=k , affinity="euclidean",linkage="ward")
#fit the model on the dataset
HClustering.fit(X)
#accuracy of the model
sm.accuracy_score(target,HClustering.labels_)


# #  Cohen Kappa Accurucy for Seed dataset (HCA)

# In[36]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[45]:


Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


# # Logistic Regression Accuracy 

# In[46]:


#Logistic Regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Logistic Regression :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for LR

# In[47]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # K-Nearest Neighbors Accuracy

# In[48]:


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("K Nearest Neighbors :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for KNN

# In[49]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Support Vector Machine Accuracy

# In[50]:


#Support Vector Machine
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Support Vector Machine:")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for SVM

# In[51]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Gaussian Naive Bayes Accuracy

# In[61]:


#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
Xtrain, Xtest, y_train, y_test = train_test_split(X, y)
classifier = GaussianNB()
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("Gaussian Naive Bayes :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for GNB

# In[62]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Decision Tree Classifier Accuracy

# In[63]:


#Decision Tree Classifier
from sklearn.model_selection import train_test_split
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

classifier = DT(criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Decision Tree Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for DTC

# In[64]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# # Random Forest Classifier Accuracy

# In[67]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier as RF
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
classifier = RF(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(Xtrain,y_train)
y_pred = classifier.predict(Xtest)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest Classifier :")
print("Accuracy = ", accuracy)
print(cm)


# # Cohen Kappa Accuracy for RFC

# In[68]:


from sklearn.metrics import cohen_kappa_score
cluster = cohen_kappa_score(y_test, y_pred)
cluster


# In[ ]:




