
##################################################################################################################################################################################
## Objective: Machine learning on hand-written digits recognition with logistic regression model. In this task, we just do the binary classification between digits '2' and '3'. #                                                                                                       #
## Data source: zip.train.csv  and zip.test.csv                                                                                                                                  #                             
## Please install "lattice" package: install.packages("lattice") for trellis visuals                                                                                             #
##################################################################################################################################################################################


## DATA EXPLORATION
## import both zip.train and zip.test to R 
## Be sure that you set working dirctory dataset folder
zip.train <- read.csv("C:/Users/muckam/Desktop/DataScienceBootcamp/Datasets/Zip/zip.train.csv", header=FALSE)
zip.test <- read.csv("C:/Users/muckam/Desktop/DataScienceBootcamp/Datasets/Zip/zip.test.csv", header=FALSE)
## check dataset dimensions
dim(zip.train)
dim(zip.test)
## check the first few lines of zip.train dataset 
## V1 represents the number, V2 -> V257 are the gray levels of all pixels
head(zip.train)
## visualize data
library(lattice)
levelplot(matrix(zip.train[5,2:257],nrow=16, byrow=TRUE))

## BUILD MODEL
## retain the rows with labels "2" and "3" in training and testing datasets
zip.train <- subset(zip.train, zip.train$V1==2 | zip.train$V1==3)
zip.test <- subset(zip.test, zip.test$V1==2 | zip.test$V1==3)
## convert V1 (response) to factor for the training & testing datasets
zip.train[,1] <- as.factor(zip.train[,1])
zip.test[,1] <- as.factor(zip.test[,1])
## fit a logistic regression model with the training dataset
zip.glm.model <- glm(formula=V1 ~ ., data=zip.train, family = "binomial", maxit=200)
### extract logistic regression model summary
summary(zip.glm.model)

## MODEL EVALUATION
## to predict using logistic regression model, probablilities obtained
zip.glm.predictions <- predict(zip.glm.model, zip.test, type="response")
## Look at probability output
head(zip.glm.predictions)
## assign labels with decision rule, >0.5= "2", <0.5="3"
zip.glm.predictions.rd <- ifelse(zip.glm.predictions >= 0.5, "3", "2")
## calculate the confusion matrix
zip.glm.confusion <- table(zip.glm.predictions.rd, zip.test[,1])
print(zip.glm.confusion)
## calculate the accuracy, precision, recall, F1
zip.glm.accuracy <- sum(diag(zip.glm.confusion)) / sum(zip.glm.confusion)
print(zip.glm.accuracy)

zip.glm.precision <- zip.glm.confusion[2,2] / sum(zip.glm.confusion[2,])
print(zip.glm.precision)

zip.glm.recall <- zip.glm.confusion[2,2] / sum(zip.glm.confusion[,2])
print(zip.glm.recall)

zip.glm.F1 <- 2 * zip.glm.precision * zip.glm.recall / (zip.glm.precision + zip.glm.recall)
print(zip.glm.F1)

## extract out a row with a wrong prediction using the 50% threshold
zip.glm.prediction.matrix <- cbind(zip.glm.predictions.rd, zip.test[,1])
zip.glm.prediction.wrong <- subset(zip.glm.prediction.matrix, zip.test[,1] != zip.glm.predictions)
head(zip.glm.prediction.wrong)

## visualize one of the wrong prediction from logistic regression
levelplot(matrix(zip.test[161,2:257],nrow=16, byrow=TRUE))





