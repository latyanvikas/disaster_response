# Disaster Response Pipeline Project for Udacity Nanodegree

![Intro Pic](screenshots/header.png)


## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
  	2. [Project Structure](#structure)
	3. [Installing](#installation)
	4. [Instructions](#Instructions)

5. [Acknowledgement](#acknowledgement)
6. [Screenshots](#screenshots)

<a name="descripton"></a>
## Description

I have build this project as pert of my data science nanodegree. It contains messages from realtime disaster response event system. It makes uses of NLP (Natural language processing) model to categorize the user's response messages in real time.

It contains following sections

1) ETL pipeline - Here we are processing the data from source, transforming it ML models and saving on a sqlite database
2) ML pipeline - Here we are testing and training the NLP model
3) Webapp -deploying the model on a webapp to categorize the response on a real time.

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

<a name="structure"></a>
### Project Structure

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # message categories data to process
|- disaster_messages.csv # Message data to process
|- ETL Pipeline Preparation.ipynb - step by step details as how ETL piepline is built
|- process_data.py - This will build the ETL pipeline
|- disasterresponse.db # database to save clean data to
models
|- train_classifier.py - We are building the NLP model pipeline to train and test the model
|- classifier.pkl # Saving the model pickle file
|- ML Pipeline Preparation.ipyn - step by step details about the machine learning pipeline
screenshotes - It contains screenshot from the web app
README.md - Information about the project

<a name="installation"></a>
### Installing
To clone the git repository:
```
git clone https://github.com/latyanvikas/disaster_response_vk.git
```

<a name="Instructions"></a>
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disasterresponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disasterresponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://127.0.0.1:8000/

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program & structured code build the pipeline


