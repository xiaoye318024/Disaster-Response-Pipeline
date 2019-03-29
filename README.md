# Disaster-Response-Pipeline
A Udacity Data Scientist Nanodegree Project

### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)

## Installation <a name="installation"></a>
Beyond the Anaconda distribution of Python 3, the following packages need to be installed for nltk:
* punkt
* wordnet
* averaged_perceptron_tagger

## Project Motivation<a name="motivation"></a>
This project: Disaster Response Pipeline, uses real messages provided by Figure 8 which were sent during natural disasters either via social media or directly to disaster response organizations.

First I built an ETL pipeline that combines & processes message and category data from csv files, and output the processed data into a SQLlite database.

Next I built a machine learning pipeline for model training, fine tuning and finally, save into a pickle file for APP use.

The last step is to build an APP to extract data from ETL pipeline, provide data visualizations using plotly and display the visuallizations on the web page. The APP also provides a search function, which use the model created from ML Pipeline to classify new messages for 36 categories.

This APP with Machine Learning is quite useful to help Rescue Agency to understand and prioritize the rescure tasks base on the analyze result of whether these messsages are relevant. In the emergency suitation such as nature disasters, this APP can be used to filter out
messages that matter, and find basic methods such as using key word searches to provide trivial results. 

In this project I practiced the skills in ETL pipelines, natural language processing, and machine learning pipelines to build up an usefull app which really has real world significance.

## File Descriptions <a name="files"></a>
There are 2 notebooks available here to showcase work related to the above questions. One for ETL Pipeline to process the data, and the other is ML Pipeline to build the model. The notebooks are exploratory in searching through the data pertaining to the questions showcased by the notebook title. Markdown cells & comments were used to assist in walking through the thought process for individual steps.

- In working_directory/data:
    * process_data.py:                 ETL Pipeline Script to process data
    * ETL Pipeline Preparation.ipynb:  jupyter notebook records the progress of building the ETL Pipeline
    * disaster_messages.csv:           Input File 1, CSV file containing messages
    * disaster_categories.csv:         Input File 2, CSV file containing categories
    * DisasterResponse_Processed.db:   Output File, SQLite database, and also the input file of train_classifier.py
    
- In working_directory/models:
    * train_classifier.py:             Machine Learning pipeline Script to fit, tune, evaluate, and export the model to a Python pickle file
    * ML Pipeline Preparation.ipynb:   jupyter notebook records the progress of building the Machine Learning Pipeline
    * model.p:                         Output File, a pickle file of the trained Machine Learning Model

- In working_directory/app:
    * templates/*.html:                HTML templates for the web app.
    * run.py: Start the Python server for the web app and prepare visualizations.

### Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse_Processed.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse_Processed.db models/model.p`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>
Average overall accuracy:     94.71%
F1 score (custom definition): 93.58%

Screenshot of the APP:
![alt text](https://github.com/xiaoye318024/Disaster-Response-Pipeline/blob/master/png/DRP%20Screenshot%201.PNG?raw=true)
![alt text](https://github.com/xiaoye318024/Disaster-Response-Pipeline/blob/master/png/DRP%20Screenshot%202.PNG?raw=true)
![alt text](https://github.com/xiaoye318024/Disaster-Response-Pipeline/blob/master/png/DRP%20Screenshot%203.PNG?raw=true)
