---
title: "Machine Learning Lifecycle"
author: Srinidhi Patil
categories:
  - machine learning
tags:
  - machine learning
  - data science
  - modeling
  - ml lifecycle
  - srinidhi patil
---


## What is a machine learning lifecycle?
Every process in software engineering involves a certain number of steps or procedures to be followed in a certain order sometimes, recursively. These steps are usually referred to as the Software Engineering lifecycle. Software Engineering lifecycle steps are part of any software development process. Just as in the software engineering lifecycle, when building a machine learning product or a service machine learning engineers and data scientists follow a set of steps and processes collectively part of what is referred to as a Machine Learning lifecycle. The Machine Learning life cycle comprises steps from data collection, and model building to monitoring the deployed model performance. We will see the steps in a machine learning lifecycle in detail in this article.

## Steps in a machine learning lifecycle
1. Data Collection
2. Data Preprocessing
   * Data Cleaning
   * Exploratory Data Analysis
   * Feature Engineering
   * Feature Selection
3. Assumption Checks
4. Model Selection
5. Model Building
   * Splitting the data
   * Selecting the libraries
6. Model Validation
   * Model Testing
   * Hyperparameter Tuning
   * Interpretability and Explainability
7. Model Deployment
8. Monitoring and Improvement

## Data Collection
The data for your project may not be readily available, you will have to define the type, parameters, and structure of the data. Data can be collected from reliable sources or generated through a defined process. Generating the data may involve a manual or an automated process. Data collection has to be done carefully to ensure that society's bias does not propagate to the data and the model. 
The data may be structured(CSV, sheets, databases) or unstructured(images, audio, text files). 

## Data Preprocessing
The data collected in the first step may have to be processed before building a model. This step involves building a pipeline of different tasks that clean and modify the data as required by the model.   

#### Data Cleaning
The data collected may be spread into multiple tables or sheets which have to merge into a single sheet. Duplicate values and missing values have to be removed or imputed.

#### Exploratory Data Analysis
Exploratory Data Analysis(EDA) is a process of understanding the data through patterns, charts/graphs, spotting anomalies, finding outliers, knowing the distribution of the data, and testing hypotheses. EDA exposes the outliers in the data that may cause issues in the model’s learning and sometimes result in derailing the model; these outliers have to be dealt with before building the model. EDA spots the patterns and the distribution of data. If the data has to be modified to fix the distribution and deal with outliers then it has to be done during this step. We can establish and prove the hypotheses on the data in this step before moving to the selection of variables.

#### Feature Engineering 
Feature engineering comprises steps in which existing data is extrapolated and new more valuable features are created. Let us look at a simple example, we have a dataset of customers of a retail company and want to build a Customer Segmentation model. We have “Date of birth” as one of the columns, it may not mean anything to the model if it is used as it is. We can extrapolate “Date of birth” into a new column called “Age”. “Age” becomes an important feature in the Customer Segmentation that we will build. 

Feature Engineering can be done through a simple data extrapolation or data transformation through a statistical/mathematical function. Examples of feature engineering through statistical functions are Scaling, Normalization, Standardization, and custom functions. 

#### Feature Selection
Feature selection is the process of selecting the right variables for the model. Using all the features in the data may cause a few issues in the model such as an unnecessary hike in accuracy, increased computation, and time for building the model. Hence, it becomes necessary to select and retain the right variables while building a model. Feature selection can be done in many ways but commonly it is done using Backward Elimination, Forward Selection, Recursive feature Elimination(RFE), or Filter methods. The technique used for feature selection will be largely based on the problem being solved(like Regression of Classification) and also on the input variables. Assumption checks also play a role in feature selection, we will study it in the next section.

## Assumption Checks
Almost every machine learning algorithm has its own set of assumptions that have to be proved before building a model or by recursively building the model. For example, linear regression assumes:
1. Linearity: The relationship between the independent variable(X) and the mean of the dependent variable(Y) is linear.
2. Homoscedasticity: The variance of residual is the same for any value of X.
3. Independence: Observations are independent of each other.
4. Normality: For any fixed value of X, Y is normally distributed.

## Model Selection
Model selection is the first step in the model-building phase. Model selection is mainly based on the problem being solved. There are different types of algorithms for each task such as classification, regression, and clustering. Selecting a model becomes one of the important tasks as it requires knowledge of available candidate models. The model has to be finalized based on complexity, maintainability, performance, and interpretability. The model finalized can be selected through an iterative process and also by comparing the performance of other available candidate models. 

## Model Building
Model building is the process of developing a predictive model that will draw the relationship between independent and dependent variables from the training data. The model-building phase has a few steps, let us look at them below:

#### Splitting the data
The data has to be split into “Train” and “Test” subsets. The Train set will be used to train the model and the Test set will be used to Test the model after it has been built. Generally, the dataset is split into Train and Test in the ratio of 70:30 or 80:20. The Test set allows us to validate the performance of the model on different metrics before refinement, revisions, and deployment.  

#### Selecting the libraries
The models can be built from scratch by the developer but it takes a considerable amount of time. There are a few well-known Python libraries such as Scikit Learn, Statsmodels, etc. which can be used to build the model.

#### Fitting the model
Fitting the model involves submitting the training data, independent variables, dependent variables, and hyperparameters to the model class.

## Model Validation
The process of evaluation and verification of the model for its fit is called model validation. Model validation highlights the shortcomings of the model and allows ML engineers to refine/tune the models to overcome those shortcomings. Let us look at the processes involved in model validation.

#### Model Testing
Testing the model with the Test set will just provide us with the metrics of the model but not the exact behavior of the model. The model has to be tested in an overall manner to verify that there is no Overfitting(a scenario where the model only learns and predicts the Train set appropriately and performs badly on encountering the new records) or Underfitting(a scenario where the model does not fit the data well enough) of the model with the training data.
The model has to be run through a series of validation tests with techniques such as Cross Validation, Invariance test, Directional Expectation test, and A/B test to ensure the model is the best fit overall. 

#### Hyperparameter Tuning
Model testing lets us know the shortcomings of the model or if the model has to be improved. One of the ways to improve the model fit is by tuning the hyperparameters of the model. Simply put, hyperparameters are the arguments that we provide the model during its learning phase that define the path for its learning. The model is trained iteratively with different hyperparameter values to check if it produces the best fit model. Hyperparameter tuning is done through techniques such as Grid-Search and Random-Search.  

#### Interpretability and Explainability
When the model makes a prediction, it should be explainable as to why the model made that prediction and the reasoning behind it. The models can not be trusted just on the basis of accuracy or other metrics as the model may be behaving in a certain way because of any reason. The rationale behind model interpretability is to increase the trust in the model,  have a transparent understanding of why the model is making the decisions a particular way, and features that contribute positively or negatively towards the dependent variable. There has been progress in building algorithms or mathematical approaches that make interpretability easier but there are no major breakthroughs that make it a rapidly growing area of research. LIME, SHAP, AIF360. Fairlearn and ELI5 are some of the libraries that help make sense of the model learning and results.    

## Model Deployment
The models developed can be deployed and used in various ways, the easiest being on the web as API, container deployment, or exporting the parameters. There are a host of options available for model deployment, for example, MLBox, AWS Sagemaker, and RapidMiner. 

The area of model deployment and operations is called Machine Learning Operations(MLOps) which is similar to the Development Operations(DevOps) in the software development lifecycle.

## Monitoring and Improvement
The model has to be monitored often to keep it relevant and avoid model degradation. Models require improvements to ensure that they are always updated against problems such as data drift and concept drift. For example, the data relevant to predict house prices in the 1990s will not be relevant during the 2008 crisis and the data collected during the 2008 crisis will not be relevant today.   

## Conclusion
The Machine Learning lifecycle is a cyclical process that is followed in the ML model or ML-related software development. In this article, we have studied the brief of each process in the ML lifecycle. I hope this article was helpful please keep following algoexpplain for more such articles.
