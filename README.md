# Practical_machine_learning_project
##Course Project - Practical machine learning 
##Author - Prakhar Maini
## Key ojective 

This repo is created as the final report for the course project for "Practical Machine Learning" course offered by Coursera.  The main goal of the project is to predict the manner in which 6 participants performed some exercise as described below. This is the “classe” variable in the training set. The machine learning algorithm described below is applied to the 20 test cases available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz for automated grading. 

I have attached the code (in R) as a separate file and all the visualizations are attached separately:

## Overview of the attachments 

### R_code: the final working code outlining the analysis

### Vis 1: Correlation matrix plot 
This plot shows that for the given data set, correlated variables are ratehr less. If we wee high number of correlated variables, we can use PCA or clustering basaed dimensionality reduction techniques to remove them from the systenm and do a more rigorous analysis

### vis 2: Prediction plot of accuracy for random forest 
The accuracy for Random Forest model on the validation data set is "0.9985"

### vis 3: Prediction plot of accuracy for Decision Tree model
The accuracy for  Decision Tree Model on the validation data set is "0.7351"

### vis 4: Prediction plot of accuracy for GBM model
The accuracy for GBM model on the validation data set is "0.9881"

### Confusion Matrix(s)
The confusion matrix for each method is saved in this file

## Data load and processing 

### Data Loading
The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Description of the datasets content from the authors’ website:

“Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).
Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg)."

### Data Cleaning 
I have performed three key steps for data cleaning
1. Removing independent variables that had near zero variance as they will not really contribute in explaining the variance of dependent variable
2. Removing variables that were mainly NAs
3. Variables that were purely for identification (metadata for records)

### Correlation analysis 
Correlation analysis is key to understand if dimensionality reduction is needed for the data. Inthe present case I have not applier PCA 

## Model building 
I built RandonForest, Decision Tree and GBM based model for prediction. Random forest performed the best. (for accuracy and confusion matrix please see the visualizations)

## Prediction on the test data
Finally RandomForest was used to predict on the test dataset.
