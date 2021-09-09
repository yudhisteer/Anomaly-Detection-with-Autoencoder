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

## 3. Exploratory Data Visualization

We will then perform box-and-whisker plots to find outliers. We will not remove these outliers for now as it is these outliers that our model will need to detect. Instead we will save these data as "Anomalous Data" and use them to test our model. 

![Box and Whisker Plot](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/4.png)


For better vizualization, we will chop our data into each day of the week. We will then create a dashboard with Plotly and Dash to visualize if there are anomalies on a particular day. 
```
# Launch the application:
# Build App
app = JupyterDash(__name__)

# Create a DataFrame from the .csv file:
app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.H1(children='Dashboard for vibration - Week 1'),

        html.Div(children='''
            Data for Day 1.
        '''),
        dcc.Graph(id = 'dashboard',
                                 figure = {'data':[go.Scatter(x=df_W1D1['Time'],
                                                              y = df_W1D1['Vibration_Day_1'],
                                                               mode = 'lines' )],

                                           'layout':go.Layout(
                                               title = 'Vibration Day 1',
                                               xaxis = {'title':'Time'},
                                               yaxis = {'title': 'Vibration'}
                                           )})])])
                                           
# Run app and display result inline in the notebook
app.run_server(mode='external',port=8060)
```

On day 1 for week 1, the motor has been on for only approx. 4 hours, from 19:45 to 23:59. For the other days, it is on for for 24 hours. 

![Dashboard](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/dashboard.jpg)

We can already see an anomaly signal on Day 1 where the value decreases to below 10 ms-2 at 20:13:59 and then increases gradually to the normal value of approx. 20 ms-2. Even on Day 2 we can see some drop in the signal at noon and during the night. The good thing about this dataset is that it already has some anomalous data which we can use to test our model. However, we need to clean the dataset of these anomalous data before training our model.

## 4. Data Wrangling
We have 1440 data points in one single day. Since we will be doing a time-series analysis, we will split our waveform into equal segments of 180 datapoints. This will increase our dataset in training our model and also allow us to segment anomalous data. We will then create two dataframes separating "Normal Vibration" data from "Anomalous Vibration" data.

```
#Slicing of data based on index
df_W1D1_1 = df_W1D1[['Vibration_Day_1']].loc['1185':'1364']

#Changing Columns name
df_W1D1_1 = df_W1D1_1.rename(columns={'Vibration_Day_1': 'Vibration_Day_1_I1'})

df_W1D1_1_index = df_W1D1_1.reset_index(drop=True)
df_W1D1_1_index
```

![SEGMENT](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/Garment%20Measurement.jpg)

We can see in the Anomalous Data segment that half the signal is an anomaly and the other half is a normal signal. This will allow us to test if K-means can identify this irregularity. 


## 4. K-Means Clustering
K-means clustering is a simple and useful unsupervised learning algorithm. The goal of K-means clustering is to group similar data points into a set number (K) of groups. The algorithms does this by identifying 'centroids', which are the centers of clusters, and then allocating data points to the nearest cluster.

How to know number of clusters(K)?
The technique to determine K, the number of clusters, is called the **elbow method**. The idea is to run k-means clustering for a range of clusters k (1 to 10) and for each value, we are calculating the sum of squared distances from each point to its assigned center(distortions).

We’ll plot:
- values for K on the horizontal axis
- the distortion on the Y axis (the values calculated with the cost function).

```
# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(x)
    kmeanModel.fit(x)
    distortions.append(sum(np.min(cdist(x, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / x.shape[0])

# Plot the elbow
figure(figsize=(8, 6), dpi=80)
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```

This result in:

![SEGMENT](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/elbow.png)

In the above plot, the elbow is at k=2 (i.e. Sum of squared distances falls suddenly) indicating the optimal k for this dataset is 2.

We will then pass our anomalous data into our K-means model with K = 2 to check how well it is classified.
```
data_with_clusters = df.copy()
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=80)
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(np.arange(180),data_with_clusters['Anomalous'],data_with_clusters['Normal'], c=data_with_clusters['Clusters'],cmap='rainbow')
plt.title("Kmeans Clustering with K = 2")
plt.show()
```
![ds](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/kmeans.png)

Our simple model did a pretty good job!
The red dots are classified as anomaly and the purple ones as normal data.
Now we have a way to classify abnormal data in a simple one dimensional space. We can adjust the percentile_treshold variable to see how that impacts the number of false positives and false negatives.

One of the key problems that K-Means has is that as the data set increases, or the number of observations we have in your collection actually increases, K-Means starts to fall apart. Especially when we have high dimensional data, it does really poor.

We will move on to work with autoencoder where our semi-supervised model will be trained on the normal rhythms only, then use it to reconstruct all the data. Our hypothesis is that the abnormal rhythms will have higher reconstruction error. We will then classify a rhythm as an anomaly if the reconstruction error surpasses a fixed threshold.

## 5. Build Autoencoder Model
With anomaly detection, the problem is that we have a lot of data for the normal behavior and those really important, yet rare events are the ones that we care about the most. We have to come to terms with the fact that we don't know what we don't know. We don't know the 1,000 ways in which the machine might fail. But there is one thing that we do know really well. We do know how the machine is supposed to work.
We don't know the various anomalies that we might see, but when everything is good, we know what that pattern is supposed to be. We can exploit the fact that we have lots of normal data, forgetting the anomaly itself. What we want to do is, effectively, we want to build in an ideal class, a neural network that can act as an identity function. In other words, it's supposed to be able to take an input and regenerate the exact same input.

Because we have lots of normal data, we can certainly architect a neural network to be able to take a set of inputs on the left-hand side and split them out on the right-hand side. That's where the interesting aspects come in. When you have a lot of input signals that are coming in, there's a lot of noise in these signals. What we really want to do, is we want to compress that input signal down to its core fundamental elements. In other words, you want to eliminate the noise that's in the signal and really get to the essence of that signal. And that's where the autoencoder comes to bear.

In simple terms, create an encoder network on the left-hand side, whose job is to take this high dimensional data that's coming in on the input neurons and then compress it down into this core compressed latent space representation, which we can then deconstruct via the decoder to reconstruct the original signal. It is possible to get a reconstructed network that minimizes this delta between what the input signal looks like and what that output signal needs to look like. In effect, you want to minimize the **reconstruction loss**. 

It's not going to be able to generate that original image, thus, when we do the difference between the two of them, it's going to be significantly larger, because it only knows how to reconstruct the normal data that it's been trained on. So when it sees an abnormal data, the error is going to be large. Hence, it can be applied to anomaly detection by training only on normal data and then using the reconstruction error to determine if a new input is similar to the normal training data.

### Types of autoencoders:
* Fully Connected Autoencoder
* Convolutional Autoencoder
* LSTM Autoencoders
* Variational Autoencoders

Because we are dealing we a one dimensional problem, I chose the a deep fully-connected autoencoder which has multiple fully connected layers with some small embedding in the middle. They are the most basic form of an autoencoder and are simple to implement and deploy.

![autoencoder](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/autoencoder.jpg)

We will concatenante our split normal data into one dataframe and transpose it such that each row contains 180 data points. We will label our data as ```1``` as normal data and ```0``` as anomaly. 

```
# Concatanate Data
df_concat =  pd.concat([df_W1D2_1_index, df_W1D2_2_index, df_W1D2_3_index, df_W1D2_4_index, df_W1D2_5_index, df_W1D2_6_index, df_W1D2_7_index, df_W1D2_8_index, df_W1D1_1_index], axis=1)

#Transpose Concatanated Data
df_concat_Transposed = df_W1D2_concat.T 

#Resetting Index
df_concat_Transposed = df_concat_Transposed.reset_index(drop=True)

#Add Labels
df_concat =  pd.concat([df_concat_Transposed, Labels], axis=1)
```

We will then split our data with ```train_test_split()```

```
# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

# Splitting data into training and test set
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.4, random_state=21)
```

We normalize the data to ```[0,1]``` to improve training accuracy.

```
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
```

This time we are mixing in some of the anomaly data into the training set.

10% of the training data will contain anomalies. Since the majority of the training data is still normal data, we can still train a high performing model as long as we don't overfit to the training data. This is to reflect a truely unsupervised scenario. Since the majority of our training data is still normal, we are still able to train a high performing model as long as we don't overfit to our training data. Train the autoencoder using only the normal rhythms, which are labeled in this dataset as 1. Separate the normal rhythms from the abnormal rhythms.

```
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

portion_of_anomaly_in_training = 0.1 #10% of training data will be anomalies
end_size = int(len(normal_train_data)/(10-portion_of_anomaly_in_training*10))
combined_train_data = np.append(normal_train_data, anomalous_test_data[:end_size], axis=0)
combined_train_data.shape
```

## 6. Picking an Embedding to Build the Model

For the encoder part, I chose a layer with ```32``` neurons and one with ```16``` neurons with a ```relu``` activation function. The smallest layer has an embedding size of ```2```. The decoder, being the inverse of the encoder, has a firs tlayer of ```16``` neurons then a layer of ```32``` neurons. And a last layer of ```180``` neurons - since we split each data into 180 data points - with a ```sigmoid``` activation function.
```
EMBEDDING_SIZE = 2

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(8, activation="relu"),
      
      layers.Dense(EMBEDDING_SIZE, activation="relu")]) # Smallest Layer Defined Here
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(8, activation="relu"),
      layers.Dense(140, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
print("Chosen Embedding Size: ", EMBEDDING_SIZE)
```

## 7. Train the model
The autoencoder is trained using only the normal data, but is evaluated using the full test set.

```
history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=100, 
          batch_size=2,
          validation_data=(test_data, test_data),
          shuffle=True)
```

```
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
```
![loss](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/loss.png)


## 8. Evaluate Training
We will soon classify a vibrational data as anomalous if the reconstruction error is greater than one standard deviation from the normal training examples. First, let's plot a normal data point from the training set, the reconstruction after it's encoded and decoded by the autoencoder, and the reconstruction error.

```
encoded_imgs = autoencoder.encoder(anomalous_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
figure(figsize=(10, 6), dpi=80)
plt.plot( anomalous_train_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(180), decoded_imgs[0],  anomalous_train_data[0], color='lightcoral' )
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title('Detecting Anomalies')
plt.show()  
```

![data](https://github.com/yudhisteer/Anomaly-Detection-with-Autoencoder/blob/main/Plots/detecting_anomalies.png)

We can clearly see the divergence from the reconstruction pattern of the autoencoder after being trained on normal data points and the actual anomalous data. The shaded region shows the Aread Under Curve(AUC) between the two signals which we will use as a threshold to classify future anomalies. 

## 9.  ROC and AUC Metrics
The Receiver Operating Characteristic (ROC) plots allows us to visualize the tradeoff between predicting anomalies as normal (false positives) and predicting normal data as an anomaly (false negative). Normal rhythms are labeled as 1 in this dataset but we have to flip them here to match the ROC curves expectations.

The ROC plot now has threshold values plotted on their corrispoinding points on the curve to aid in selecting a theshold for the application.

```
reconstructions = autoencoder(test_data)
loss = tf.keras.losses.mae(reconstructions, test_data)
fpr = []
tpr = []
#the test labels are flipped to match how the roc_curve function expects them.
flipped_labels = 1-test_labels 
fpr, tpr, thresholds = roc_curve(flipped_labels, loss)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve ')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

# plot some thresholds
thresholds_every=20
thresholdsLength = len(thresholds)
colorMap=plt.get_cmap('jet', thresholdsLength)
for i in range(0, thresholdsLength, thresholds_every):
  threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
  plt.scatter(fpr[i], tpr[i], c='black')
  plt.text(fpr[i] - 0.03, tpr[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 15});

plt.show()
```

()
## 10.  Picking a Threshold to Detect Anomalies


## Limitations

## Conclusion

























