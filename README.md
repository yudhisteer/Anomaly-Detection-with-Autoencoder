# Anomaly-Detection-with-Autoencoder

In an industry, time equals money. Each minute a machine breaks down, it affects the workflow, production stops, revenue dwindles. It is imperative for the company to have a contingency plan such as having a backup machine to prevent the process coming to a standstill or another solution which is even better is to know in advance which machine will breakdown and when so that corrective measures can be applied. In this project I analyzed the vibrational data of AC motors used in the Dyeing Department of a Textile company I was working and devised an anomaly detection algorithm that would help the production engineers to have a proper planning in case of breakdown. With such a system, the aim is to reduce financial loss that incurs due to unanticipated machine breakdown. 

## Research Questions
Before starting the project, I brainstormed about a few research questions which I wanted to answer in order to come to the solution. 

* Is it possible in the first place to predict if a machine will fail?
* If we can, on what parameters should we be able to make this prediction?
* What makes a machine breakdown and can we stop it from malfunctioning originally?
* Can this prediction system be scaled up such that we predict possible failures of other machines?
* What metric will we use to posit that a machine will break down in the future?
* How much in advance can we predict if a machine will break down? Days? Weeks? Months?

## Methods


## Abstract


## Dataset(s)
The data is generated from IoT sensors and sent to a server to be stored in a database. The system captures the data on a 3 minutes interval for 24 hours and the data is sent to the database on a weekly basis.

* Raw_Data_Week.csv: contains 12 features with the timestamp for a particular week.

## Plan of Action

1. Load Dataset
2. Data Visualization
3. Data Cleaning
4. K-Means Clustering
5. Build Autoencoder Model
6. Picking an Embedding to Build the Model
7. Train the model
8. Evaluate Training
9.  ROC and AUC Metrics
10.  Picking a Threshold to Detect Anomalies

## 1. Load Dataset
The dataset as shown below has 12 features with the timestamp. However, a lot of these features are redundant as their values are constant. To start simple, I selected only the date,  timestamp and the column "VibrationPump2_Output". 

![Raw Data](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/rawdata.jpg)

We also load the necessary libraries:
```
!pip install jupyter-dash
import plotly.express as px
from plotly.offline import iplot
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
!pip install dash
import dash
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
```

## 2. Data Cleaning
We start by loading our dataset for the first week. We rename the column for vibration and remove the zero values in the latter. The zero values signify that the pump was off. We do not want our model to think that the zero values are anomalies hence, we discard them. We analyze the number of data points with which we will work and print the first 5 rows.
```
# Read Dataset of Week 1
df_with_zeros = pd.read_csv('Raw_Data_Week1.csv')

#Rename column for Vibration data
df_with_zeros = df_with_zeros.rename(columns={'Vibration_Week1': 'VibrationPump2_Output'}

#Drop vibration values of zero
df_without_zeros = df_with_zeros[df_with_zeros['Vibration_Week1'] != 0]

#Print number of datapoints
len(df_without_zeros)

#Checking type of Data
print(df_without_zeros.dtypes)

#Print first 5 rows of dataset
print(df_without_zeros.head())

```
Since we will do a time-series analysis it is important we set our date format in ```datetime64[ns]``` using ```pd.to_datetime```. 

We will do some exploratory data visualization first and proceed with more data manipulation thereafter. 

## 3. Data Visualization

We will then perform box-and-whisker plots to find outliers. We will not remove these outliers for now as it is these outliers that our model will need to detect. Instead we will save these data as "Anomalous Data" and use them to test our model. 

![Box and Whisker Plot](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/4.png)



![Dashboard](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/dashboard.jpg)



## 4. K-Means Clustering
## 5. Build Autoencoder Model
## 6. Picking an Embedding to Build the Model
## 7. Train the model
## 8. Evaluate Training
## 9.  ROC and AUC Metrics
## 10.  Picking a Threshold to Detect Anomalies


## Limitations

## Conclusion

























