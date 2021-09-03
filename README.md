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

## Limitations

## Conclusion

























