# Practical_machine_learning_project
##Course Project - Practical machine learning 
##Author - Prakhar Maini
## Key ojective 

This repo is created as the final report for the course project for "Practical Machine Learning" course offered by Coursera.  The main goal of the project is to predict the manner in which 6 participants performed some exercise as described below. This is the “classe” variable in the training set. The machine learning algorithm described below is applied to the 20 test cases available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz for automated grading. 

I have attached the code (in R) as a separate file and all the visualizations are attached separately:

I have used cross validation for the RF model and the Out of bag error for the best model (RF) was 

## Overview of the attachments 

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

### R_code: the final working code outlining the analysis

library(knitr)

library(caret)

library(rpart)

library(rpart.plot)

library(rattle)

library(randomForest)

library(corrplot)

set.seed(42)

########################### downloading train and test data ####################################
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" # setting the url for training data

UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" # setting the url for test data

training <- read.csv(url(UrlTrain)) # downlaoding training data

testing  <- read.csv(url(UrlTest)) # downlaoding test data

########################## creating data partitions for validation (split 70 / 30) ############
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)

TrainSet <- training[inTrain, ] # training set

TestSet  <- training[-inTrain, ] # validation set 

dim(TrainSet)

dim(TestSet)

############################ data preprocessing ##############################################

NZV <- nearZeroVar(TrainSet) # removing variable with near zero variance

TrainSet <- TrainSet[, -NZV] 

TestSet  <- TestSet[, -NZV]

dim(TrainSet)

dim(TestSet)

AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95 # removing variables and are mostly NAs

TrainSet <- TrainSet[, AllNA==FALSE]

TestSet  <- TestSet[, AllNA==FALSE]

dim(TrainSet)

dim(TestSet)

TrainSet <- TrainSet[, -(1:5)] # removing the identification only variables in column 1 to 5 

TestSet  <- TestSet[, -(1:5)]

dim(TrainSet)

dim(TestSet)

########################### creating correlation matrix ######################################

corMatrix <- cor(TrainSet[, -54])

corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))

################################ model building ##############################################

################################### Random Forest model #######################

set.seed(42)

controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)

modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf", trControl=controlRF) # building model on the training data

modFitRandForest$finalModel 

predictRandForest <- predict(modFitRandForest, newdata=TestSet) # getting predictions on the validation data

confMatRandForest <- confusionMatrix(predictRandForest, TestSet$classe)

confMatRandForest

plot(confMatRandForest$table, col = confMatRandForest$byClass, main = paste("Random Forest - Accuracy =", round(confMatRandForest$overall['Accuracy'], 4)))

##################################### Decision Tree Model #######################

set.seed(42)

modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class") # building model on the training data

fancyRpartPlot(modFitDecTree)

predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class") # getting predictions on the validation data

confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)

confMatDecTree

plot(confMatDecTree$table, col = confMatDecTree$byClass, main = paste("Decision Tree - Accuracy =", round(confMatDecTree$overall['Accuracy'], 4)))

######################################## GBM Model ################################

set.seed(42)

controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm", trControl = controlGBM, verbose = FALSE) # building model on the training data

modFitGBM$finalModel

predictGBM <- predict(modFitGBM, newdata=TestSet) # getting predictions on the validation data

confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)

confMatGBM

plot(confMatGBM$table, col = confMatGBM$byClass, main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))

####################### Applying the chosen model (Random Forest) to the test data ##########################

predictTEST <- predict(modFitRandForest, newdata=testing)

predictTEST
