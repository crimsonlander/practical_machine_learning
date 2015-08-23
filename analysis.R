#' ---
#' title: "Practical Machine Learning course project."
#' author: "Denis Kuzminykh"
#' date: "August 23, 2015"
#' ---
#' 
#' ###Preface:
#' 
#' In this project I have done human activity prediction using 3-layer neural network with one hidden layer.
#' No exploratory analysis has been done; I have cleared the data, preprocessed it with PCA and tuned 
#' neural network to predict answers with high accuracy.  
#' 
#' ####Feature selection: 
#' 
#' I have used all the data except date, near zero variables and variables with missing values. In this 
#' data set variables either don't have missing values or have almost all values missing, so it is 
#' reasonable to drop the last. Alter this step only 59 features left.
#' 
#' ####Preprocessing:
#' 
#' Person names have been turned into dummy variables. All the variables have been centered by the mean and 
#' scaled by the standard deviation. Then 37 principal components have been extracted retaining 99% of 
#' variance.
#' 
#' ####Training:
#' 
#' I have used Feed-Forward Neural Network with 1 hidden layer of 28 neurons from nnet package.
#' Weight decay have been used for regularization.
#'  
#' ###The analysis:

library(caret); library(nnet)

set.seed(100)

#' Get the data:
d = read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", method="libcurl"))

#' Separate labels from features:
labels = d[,160]

#' Drop row numbers, date, near zero variance features and labels from the data:
nzv <- nearZeroVar(d)
d <- d[, -c(1, 3, 4, 5, 160, nzv)]
#' Note: there might be a relation between time and action. I decided not to use it because it
#' would not be very useful in real life HAR problems.

#' Turn user_name into a set of dummy variables.
dv = dummyVars(~ user_name, data = d)
d <- cbind(d, predict(dv, d))
d$user_name <- NULL

#' There are lot of columns with lots of NAs and very little data.
not_na = !is.na(colSums(d))
d <-d[, not_na]

#' Partition the data into trainning, test and validation sets.
inVal <- createDataPartition(y=labels, p=0.05, list=FALSE)

restFeatures <- d[-inVal,]
restLabels <- labels[-inVal]

valFeatures <- d[inVal,]
valLabels <-labels[inVal]

inTrain <- createDataPartition(y=labels, p=0.8, list=FALSE)

trFeatures <- restFeatures[inTrain,]
trLabels <- restLabels[inTrain]

teFeatures <- restFeatures[-inTrain,]
teLabels <-restLabels[-inTrain]

rm(restFeatures, restLabels)

#' Preprocessing step. Principal component analysis here effectively reduces amunt of features.
#' PCA thresh value is important and significantly affects accuracy. 
#' 0.99 gives much better accuracy than default 0.95. (approximately 0.97 vs 0.92 accuracy)
pc_selector <- preProcess(trFeatures, method = c("center", "scale", "pca"), thresh = 0.99)

trPC <- predict(pc_selector, trFeatures)
tePC <- predict(pc_selector, teFeatures)
valPC <- predict(pc_selector, valFeatures)

#' Train the 3-layer neural network with 1 hidden layer with 28 units in it. Can take 10 minutes.
#' Weight decay helps to regularize the network and reduce overfitting.
nnet_model = nnet(trLabels ~ ., data = trPC, size=28, maxit = 5000, decay = 0.1, MaxNWts=2000)

#' ####The trainning results:
predictions = as.factor(max.col(predict(nnet_model, tePC)))
levels(predictions) = levels(labels)
confusionMatrix(teLabels, predictions)

#' ###Validation set. 
#' It was used to make sure nnet parameters are not overfitted.
predictions = as.factor(max.col(predict(nnet_model, valPC)))
levels(predictions) = levels(labels)
confusionMatrix(valLabels, predictions)

#' ####Out of sample error:
#'
#' I have done some kind of bootstrapping by running this script with different seed values: 
#' 
#' 100 : 0.9777 accuracy, 
#' 101 : 0.9888 accuracy, 
#' 102 : 0.9742 accuracy
#' 
#' It suggests reasonable expectation that error rate should be less than 4%
#' Doing more serous cross-validation doesn't seem reasonable to me because it
#' consumes lots of time and shows high accuracy. I think there might be a little 
#' overfitting, but this can't be shown by cross-validation using this data.
#'
#'
#' ####Test
#' Now, lets go to the real test data.

d <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", method="libcurl"))

#' Do the same feature selection and preprocessing steps.
d <- d[, -c(1, 3, 4, 5, 160, nzv)]

dv = dummyVars(~ user_name, data = d)
d <- cbind(d, predict(dv, d))
d$user_name <- NULL
d <-d[, not_na]
PC <- predict(pc_selector, d)

#' Apply trained network to predict answers.
answers = as.factor(max.col(predict(nnet_model, PC)))
levels(answers) = levels(labels)

answers

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

dir.create("answers", showWarnings = FALSE)
setwd("answers")
pml_write_files(answers)
setwd("..")

#' Hooray! 19 out of 20.
