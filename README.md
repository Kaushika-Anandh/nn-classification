Exp.No : 02 
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
Date : 28.08.2023
<br>
# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

-  automobile company has plans to enter new markets with their existing products.
-  After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.
- In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ).
- Then, they performed segmented outreach and communication for a different segment of customers.
- This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.
- You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<p align="center">
<img src="https://github.com/Kaushika-Anandh/nn-classification/blob/main/nn.png" width="650" height="400">
</p>

<br>

## DESIGN STEPS

- **Step 1:** Import the necessary packages & modules
- **Step 2:** Load and read the dataset
- **Step 3:** Perform pre processing and clean the dataset
- **Step 4:** Encode categorical value into numerical values using ordinal/label/one hot encoder modules
- **Step 5:** Split and Scale the data to training and testing
- **Step 6:** Train the data using Dense module in tensorflow
- **Step 7:** Gather the trainling loss and classification metrics


## PROGRAM
> Developed by: Kaushika A <br>
> Register no: 212221230048

**importing packages**
```python
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
```

**loading the dataset**
```python
df_initial = pd.read_csv('customers.csv')
```

**EDA**
```python
df_initial.columns
df_initial.dtypes
df_initial.shape
df_initial.isnull().sum()
```

**data cleaning**
```python
df_cleaned = df_initial.dropna(axis=0)
df_cleaned.isnull().sum()
```

**EDA before Data Encoding**
```python
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()
df_cleaned['Graduated'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Var_1'].unique()
df_cleaned['Segmentation'].unique()
```

**data encoding**
```python
categories_lst=[['Male', 'Female'],
            ['No', 'Yes'],
            ['No', 'Yes'],
            ['Healthcare', 'Engineer', 'Lawyer', 'Artist',
            'Doctor','Homemaker', 'Entertainment', 'Marketing', 'Executive'],
            ['Low', 'High', 'Average']]
enc = OrdinalEncoder(categories=categories_lst)
df1=df_cleaned.copy()
df1[['Gender','Ever_Married',
     'Graduated','Profession',
     'Spending_Score']] = enc.fit_transform(df1[['Gender','Ever_Married',
                                                 'Graduated','Profession',
                                                 'Spending_Score']])
df1.dtypes

le = LabelEncoder()
df1['Segmentation'] = le.fit_transform(df1['Segmentation'])
df1.dtypes

df1.describe()
df1['Segmentation'].unique()
X = df1[['Gender','Ever_Married','Age','Graduated',
         'Profession','Work_Experience','Spending_Score',
         'Family_Size']].values

y1=df1[['Segmentation']].values
oh_enc = OneHotEncoder()
oh_enc.fit(y1)
y1.shape
y=oh_enc.transform(y1).toarray()
y.shape

y1[0]
y[0]
X.shape
```

**splitting the data**
```python
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
```

**scaling the data**
```python
scaler = MinMaxScaler()
scaler.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
```

**Creating the model**
```python
ai_brain = Sequential([
    Dense(8,input_shape=(8,)),
    Dense(16, activation ='relu'),
    Dense(16),
    Dense(8, activation ='relu'),
    Dense(4,activation='softmax')
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2)
```

**running the model**
```python
ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test))
```

**getting metrics**
```python
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
```

**plotting loss & accuracy**
```python
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
```
```python
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
```

**classification metrics**
```python
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
```

**Prediction for a single input**
```python
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```


## Dataset Information

<img src="https://github.com/Kaushika-Anandh/nn-classification/blob/main/1.PNG" width="550" height="300">

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img src="https://github.com/Kaushika-Anandh/nn-classification/blob/main/2.png" width="350" height="250">

### Classification Report

<img src="https://github.com/Kaushika-Anandh/nn-classification/blob/main/3.PNG" width="400" height="150">

### Confusion Matrix

<img src="https://github.com/Kaushika-Anandh/nn-classification/blob/main/4.PNG" width="120" height="90">

### New Sample Data Prediction

<img src="https://github.com/Kaushika-Anandh/nn-classification/blob/main/5.PNG" width="450" height="120">

## RESULT
Thus a Neural Network Classification Model is created and executed successfully
