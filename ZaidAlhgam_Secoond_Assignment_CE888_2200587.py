#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)

# 1. Import the Important Libraries 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE  # Oversample and plot imbalanced dataset with SMOTE
warnings.filterwarnings('ignore')


# 2. Load and Merge All CSVs into one Dataframe 

# In[3]:


Data_set = pd.DataFrame()   #Our row datasets will be in this dataframe 
# We will put our preprocessed dataset into new variable.
cleaned_data = pd.read_csv(r"D:/Data_Science/Assignments/First_Assignment/Stress-Predict-Dataset-main/Stress-Predict-Dataset-main/Processed_data/Improved_All_Combined_hr_rsp_binary.csv")

#There are 35 subjects data so looping through 35 data and merging them
x = 2
while x<36:
    
    path = 'D:/Data_Science/Assignments/First_Assignment/Stress-Predict-Dataset-main/Stress-Predict-Dataset-main/Raw_data/'
    
    #Read the dataset files
    ACC = pd.read_csv(path+"S"+"{0:02d}".format(x)+"/ACC.csv")
    BVP = pd.read_csv(path+"S"+"{0:02d}".format(x)+"/BVP.csv")
    EDA = pd.read_csv(path+"S"+"{0:02d}".format(x)+"/EDA.csv")
    HR = pd.read_csv(path+"S"+"{0:02d}".format(x)+"/HR.csv")
    IBI = pd.read_csv(path+"S"+"{0:02d}".format(x)+"/IBI.csv")
    TEMP = pd.read_csv(path+"S"+"{0:02d}".format(x)+"/TEMP.csv")
    
    # Drop the first row which contain the time of getting the values in each csv file.  
    
    ACC = ACC.drop(0).reset_index(drop = True)
    BVP = BVP.drop(0).reset_index(drop = True)
    EDA = EDA.drop(0).reset_index(drop = True)
    IBI = IBI.drop(0).reset_index(drop = True)
    TEMP = TEMP.drop(0).reset_index(drop = True)
    HR = HR.drop(0).reset_index(drop = True)
    
    # The column names starts with the start time+ 1 as a string which can be used to map the time of the data
    
    ACC['Time(sec)'] = int(float(ACC.columns[0])) - 1
    BVP['Time(sec)'] = int(float(BVP.columns[0])) - 1
    EDA['Time(sec)'] = int(float(EDA.columns[0])) - 1
    HR['Time(sec)'] = int(float(HR.columns[0])) - 1
    IBI['Time(sec)'] = int(float(IBI.columns[0])) - 1
    TEMP['Time(sec)'] = int(float(TEMP.columns[0])) - 1
    
    # Count the collecting data time for each row (each 1 Sec) by adding the number of the index of specific row.
    
    ACC['Time(sec)'] = ACC['Time(sec)']+ACC.index
    BVP['Time(sec)'] = BVP['Time(sec)']+BVP.index
    EDA['Time(sec)'] = EDA['Time(sec)']+EDA.index
    HR['Time(sec)'] = HR['Time(sec)']+HR.index
    IBI['Time(sec)'] = IBI['Time(sec)']+IBI.index
    TEMP['Time(sec)'] = TEMP['Time(sec)']+TEMP.index
    
    #Rename columns according to the readme file of the dataset.
    
    ACC = ACC.rename({ACC.columns[0]:'ACC_x',ACC.columns[1]:'ACC_y',ACC.columns[2]:'ACC_z'},axis = 1)
    BVP = BVP.rename({BVP.columns[0]:'BVP'},axis = 1)
    EDA = EDA.rename({EDA.columns[0]:'EDA'},axis = 1)
    HR = HR.rename({HR.columns[0]:'HR'},axis = 1)
    IBI = IBI.rename({IBI.columns[0]:'IBI0',IBI.columns[1]:'IBI1'},axis = 1)
    TEMP = TEMP.rename({TEMP.columns[0]:'TEMPERATURE'},axis = 1)
    
    # Merge all files into one dataframe for the next data proccessing steps.
    # We ignored the IBI file because it has a very small data comparing with all other files and that can affect uor overall data.
    data = ACC.merge(BVP,on = 'Time(sec)',how = 'inner').merge(HR,on = 'Time(sec)',how = 'inner').merge(EDA,on = 'Time(sec)',how = 'inner').merge(TEMP,on = 'Time(sec)',how = 'inner')
    final = data.merge(cleaned_data[['Time(sec)','Label']],on = 'Time(sec)', how = 'inner')
    
    Data_set = Data_set.append(final)
    
    x += 1


# 3. Check the size and shape of the data set

# In[39]:


Data_set.shape


# 4. Review the merged dataset

# In[4]:


Data_set.head()


# 5. Present the summary of the descriptive statistics for the DataFrame.

# In[6]:


Data_set.describe()


# 6. Checking if there are some Null values

# In[8]:


Data_set.isnull().sum()


# 7. Plot the time series graph to visualise and understand the data

# In[41]:


fig, ax = plt.subplots(figsize= (25,8))
ax.plot(Data_set['Time(sec)'], Data_set.BVP)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot (BVP)')
plt.show()


# In[42]:


fig, ax = plt.subplots(figsize= (25,8))
ax.plot(Data_set['Time(sec)'], Data_set.ACC_x)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')
plt.show()


# In[43]:


fig, ax = plt.subplots(figsize= (15,8))
ax.plot(Data_set['Time(sec)'], Data_set.EDA)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')
plt.show()


# In[44]:


fig, ax = plt.subplots(figsize= (25,8))
ax.plot(Data_set['Time(sec)'], Data_set.HR)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')
plt.show()


# In[45]:


fig, ax = plt.subplots(figsize= (25,8))
ax.plot(Data_set['Time(sec)'], Data_set.TEMPERATURE)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')
plt.show()


# 8. Explore the Target and its behaviour

# In[10]:


# Let's see the classes we have and the count values
Data_set['Label'].value_counts()


# Data visualization needs to be done to undersatand the data more clearly.
# From the Bar graph, we can clearly see the distribution of the both calsses is not balanced,
# and the data set could be described to have an imbalanced class distribution.
# 
# An imbalanced class distribution can lead to biased models and poor predictions for the minority class,
# so, we need to handle this issue before butting our data for training of any ML model.

# In[12]:


plt.hist(Data_set['Label'], bins=5)
plt.xticks([0, 1])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.show()


# One approach to solve this issue is by using the Up-Sampling method with the SMOTE function

# In[25]:


X = Data_set.drop('Label',axis = 1)
y = Data_set['Label']

smote = SMOTE()
X_values,Y_values = smote.fit_resample(X,y)


# In[26]:


Final_Data_set1 = X_values
Final_Data_set1['Label'] = Y_values


# In[27]:


Final_Data_set1['Label'].value_counts()


# In[29]:


# Revisualise the new Label calsses distribution
plt.hist(Final_Data_set1['Label'], bins=5)
plt.xticks([0, 1])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Distribution of Labels')
plt.show()


# 9. Recalculate the data size after the up-sampling

# In[28]:


Final_Data_set1.shape


# 10. We need some Descriptive analyses 

# In[46]:


Final_Data_set1.describe()


# In[30]:


Final_Data_set1.head()


# 11. show the correlation between each variable 

# In[31]:


Final_Data_set1.corr().transpose()


# In[32]:


sns.heatmap(Final_Data_set1.corr().transpose(), cmap='coolwarm', annot= True)
plt.show()


# 12. Present more information by showing the distribution of all features and the target 

# In[33]:


Final_Data_set1.hist(figsize = (10,10))
plt.show()


# 13. Check if there are some Outlieres

# In[34]:


plt.boxplot(Final_Data_set1)
plt.show()


# # Normalise the cleaned Data

# Before start working on the models and evaluation, the data needs to be normalised then we can split it into train and test sets

# In[35]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(Final_Data_set1.iloc[:, :-1])
normalized_data = pd.DataFrame(normalized_data, columns=Final_Data_set1.columns[:-1])
normalized_data['Label'] = Final_Data_set1['Label']
normalized_data.shape


# # Spliting the data into train and test 

# Split the normalised data as 70% train set and 30% test set  

# In[47]:


from sklearn.model_selection import train_test_split
A = normalized_data.drop(['Label'], axis= 1)
B = normalized_data['Label']
X_train_, X_test_, y_train_, y_test_ = train_test_split(A, B, test_size=0.3, random_state=42, shuffle= False)


# # Build ML Models

# 1. Long Short-Term Memory (LSTM)

# In[37]:


# import tensorflow.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# The number of time steps and features (columns in data)
time_steps = 10
n_features = 8

# The sequences of length time_steps from data
sequences = []
labels = []
for i in range(time_steps, len(normalized_data)):
    sequences.append(normalized_data.iloc[i - time_steps:i, :-1])
    labels.append(normalized_data.iloc[i, -1])
X = np.array(sequences)
y = np.array(labels)

# split the data into training and testing sets for the LSTM model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train.shape,y_train.shape


# In[38]:


# build an LSTM model
model1 = Sequential()
model1.add(LSTM(64, input_shape=(time_steps, n_features)))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)


# Use the model to Predict the test set and evaluate its performance

# In[49]:


# evaluate the model on the testing data
y_pred_prob = model1.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")


# In[65]:


metrices = pd.DataFrame(history.history)


# Plot the Accuracies

# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
plt.plot(metrices['loss'], label='train_loss')
plt.plot(metrices['accuracy'], label='train_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()


# In[73]:


plt.plot(metrices['val_loss'], label='val_loss')
plt.plot(metrices['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()


# 2. Random Forest Classifier

# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[54]:


model2 = RandomForestClassifier().fit(X_train_, y_train_)
model2_predict = model2.predict(X_test_)


# Present and Plot the Metrics 

# In[55]:


print(classification_report(y_test_,model2_predict))


# In[76]:


from sklearn.metrics import accuracy_score
accuracy_RF = accuracy_score(y_test_, model2_predict)
plt.plot([accuracy_RF]*2, label='Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()


# In[74]:


from sklearn.metrics import classification_report
report = classification_report(y_test_,model2_predict, output_dict=True)
fig, ax = plt.subplots(figsize=(8,5))
for metric in ['precision', 'recall', 'f1-score']:
    values = [report[label][metric] for label in report.keys() if label.isdigit()]
    ax.bar([label for label in report.keys() if label.isdigit()], values, label=metric)

# add a legend and labels
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_title('Classification Report')
ax.legend()
plt.show()


# 3. Gradient Boosting Classifier

# In[56]:


from sklearn.ensemble import GradientBoostingClassifier

model3 = GradientBoostingClassifier().fit(X_train_, y_train_)
model3_predict = model3.predict(X_test_)


# Present and Plot the Metrics of the GBC model

# In[57]:


print(classification_report(y_test_,model3_predict))


# In[79]:


from sklearn.metrics import accuracy_score
accuracy_GBC = accuracy_score(y_test_, model3_predict)
plt.plot([accuracy_GBC]*2, label='Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()


# In[78]:


report = classification_report(y_test_,model3_predict, output_dict=True)
fig, ax = plt.subplots(figsize=(8,5))
for metric in ['precision', 'recall', 'f1-score']:
    values = [report[label][metric] for label in report.keys() if label.isdigit()]
    ax.bar([label for label in report.keys() if label.isdigit()], values, label=metric)

# add a legend and labels
ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_title('Classification Report')
ax.legend()
plt.show()


# # Models Accuracies comparison

# In[81]:


models = ['LSTM ', 'RandomForestClassifier', 'GradientBoostingClassifier']
accuracy = [0.8547, accuracy_RF, accuracy_GBC]
colors = ['red', 'blue', 'green']
plt.bar(models, accuracy, color=colors)
plt.title('Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()


# From above graph, it is clear that the LSTM is the best model for predicting the stress according to the used dataset 
