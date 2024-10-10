#!/usr/bin/env python
# coding: utf-8

# # Exploring malware Analysis for Jar File Security through REJAFADA Dataset

# In[9]:


# Importing Libraries


# In[33]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense


# In[13]:


data = pd.read_csv(r"C:\Users\91630\OneDrive\Desktop\A16. Jar file security through REJAFADA\REJAFADA.csv", encoding='latin-1')
data


# In[14]:


# Dataset analysis


# In[15]:


data.describe()


# In[16]:


# dataset information


# In[17]:


data.info()


# In[18]:


#checking null values


# In[19]:


data.isnull().sum()


# In[20]:


data['B'].unique().sum()


# In[21]:


sns.countplot(x = data['B'])
plt.show()


# In[22]:


#Selecting dependent and independent variable


# In[23]:


X = data.drop(['A','B'],axis = 1)
X


# In[24]:


Y = data['B']
Y


# In[25]:


# Labeling the target varible


# In[26]:


le = LabelEncoder()
Y = le.fit_transform(Y)


# In[27]:


#splitting the dataset


# In[28]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
X_train


# # ANN Model

# In[29]:


model = Sequential()


model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer with 64 units and ReLU activation
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 units and ReLU activation
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 unit for binary classification (sigmoid activation)

model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=30, batch_size=16, validation_data=(X_test, Y_test))  # Adjust epochs and batch size as needed

# Evaluate the model
accuracy = model.evaluate(X_test, Y_test)[1]
print(f'Accuracy: {accuracy*100:.2f}%')


# In[ ]:


Y_pred1 = model.predict(X_test)


# In[ ]:


#ANN Model evaluation


# In[ ]:





# In[ ]:


Y_test = Y_test.astype(int)
Y_pred1 = Y_pred1.astype(int)
Y_pred1_binary = (Y_pred1 > 0.5).astype(int)


# In[ ]:


acc = accuracy_score(Y_test,Y_pred1_binary)
print(f'accuracy_score=========>',acc)


# In[ ]:


confusion = confusion_matrix(Y_test, Y_pred1)

print("Confusion Matrix for ANN model:")
print(confusion)


# In[ ]:


report = classification_report(Y_test, Y_pred1)

print("\nClassification Report for ANN model:")
print(report)


# In[ ]:


plt.figure(figsize=(6, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['A','B'], yticklabels=['A','B'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for ANN model')
plt.show()


# # RandomForest Classifier

# In[ ]:


rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
Y_pred


# In[ ]:


# Model evaluation


# In[ ]:


acc = accuracy_score(Y_test, Y_pred)
print(f'accuracy_score=========>',acc)


# In[ ]:


confusion = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix for RandomForest Classifier:")
print(confusion)


# In[ ]:


report = classification_report(Y_test, Y_pred)

print("\nClassification Report for RandomForest Classifier:")
print(report)


# In[ ]:


plt.figure(figsize=(6, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['A','B'], yticklabels=['A','B'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for RandomForest Classifier')
plt.show()


# In[ ]:




