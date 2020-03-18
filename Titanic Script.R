#Read in Data
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)
gender_submission <- read.csv('gender_submission.csv')

#Load Packages
library(tidyverse)
library(Metrics)
library(randomForest)
library(boot)
library(caret)
library(gbm)
library(xgboost)
library(mlr)
library(data.table)

#Explore Dataset
str(train)
str(test)
head(test)
head(train)
summary(train)
summary(test)

#Combine Datasets to clean together
train$IsTrain <- TRUE
test$IsTrain <- FALSE
test$Survived <- NA

full <- rbind(train, test)
table(full$IsTrain)

#Clean Full DataSet
full[full$Embarked == '', "Embarked"]

age.predict <- randomForest(Age ~ Pclass+Sex+SibSp+Parch, data = na.omit(full), ntree = 500, mtry = 3)
full[is.na(full$Age), "Age"] <- predict(age.predict, newdata = full[is.na(full$Age), ], type = "response")

table(is.na(full$Fare))
fare.predict <- randomForest(Fare~Pclass+Sex+Parch+SibSp+Embarked+Age, data = na.omit(full), ntree = 500, mtry = 3)
full[is.na(full$Fare), "Fare"] <- predict(fare.predict, newdata = full[is.na(full$Fare),], type = "response")

#Categorical Casting
full$Pclass <- as.factor(full$Pclass)
full$Sex <- as.factor(full$Sex)
full$Embarked <- as.factor(full$Embarked)

#Split Data back into Train and Test sets
train1 <- full[full$IsTrain == TRUE, ]
test1 <- full[full$IsTrain == FALSE, ]

train1$Survived <- as.factor(train1$Survived)

#Logistic Regression Model
Logistic.Regression <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train1, 
                               family = "binomial")

summary(Logistic.Regression)
train.probs <- as.data.frame(predict(Logistic.Regression, newdata = train1, type = "response"))
train.pred <- rep("0", 891)
train.pred[train.probs > .5] = "1"
confusionMatrix(data = as.factor(train.pred), reference = train1$Survived)

#Let's try a randomForest Model
rf.model <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = train1,
                         ntree = 500, mtry = 3, nodesize = .01 * nrow(test1))

rf.train.pred <- predict(rf.model, newdata = train1, type = "response")
confusionMatrix(data = rf.train.pred, reference = train1$Survived)

#This model seems to be the most accurate so far
#Lastly, Let's try XGBoost
data.xg <- train1[,c(3,5,6,7,8,10,12)]
label.xg <- train1[,"Survived"]

test.data.xg <- test1[,c(3,5,6,7,8,10,12)]

dummy.train <- dummyVars(~Pclass+Sex+Embarked, data = data.xg)
data.train <- predict(dummy.train, data.xg)

train.set <- cbind(data.xg,data.train)
train.set <- train.set[ ,c(3,4,5,6,8:15)]
train.set <- as.matrix(train.set)
#Clean Data for xgBoost
labels <- train1$Survived
ts_labels <- test1$Survived
new_tr <- model.matrix(~.+0,data = train1[,-c(1,2,4,9,11,13),with=F])

#Let's figure out how to set the parameters
params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.3, gamma=0, max_depth = 6,
               min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv(params=params, data = train.set, nrounds = 100, nfold = 10, showsd = T, stratified = T,
                print_every_n = 10, early_stopping_rounds = 20, maximize = F)

#Build the Model
xg.model <- xgboost(data=train.set, label = as.matrix(label.xg), max_depth = 20, nrounds = 500, eta = .01,
                    verbose = 1, objective = "binary:logistic", eval_metric = "auc")

dummy.test <- dummyVars(~Pclass+Sex+Embarked, data = test.data.xg)
data.test <- predict(dummy.test, test.data.xg)
test.set <- cbind(test.data.xg, data.test)
test.set <- test.set[,c(3,4,5,6,8:15)]
test.set <- as.matrix(test.set)

train.survive.pred <- predict(xg.model, newdata = train.set)
train.pred <- as.numeric(train.survive.pred > .5)

confusionMatrix(as.factor(train.pred), reference = train1$Survived)

#The XG model seemed to work the best, so we will go with that one
test.survive.pred <- predict(xg.model, newdata = test.set)
test.pred <- as.numeric(test.survive.pred > .5)

submission <- test1$PassengerId
submission <- as.data.frame(cbind(submission, test.pred))
names(submission) <- c("PassengerId", "Survived")

log.pred <- predict(Logistic.Regression, newdata = test1, type = "response")
log.pred <- as.numeric(log.pred > .5)
logistic.submission <- test1$PassengerId
logistic.submission <- as.data.frame(cbind(logistic.submission, log.pred))
names(logistic.submission) <- c("PassengerId", "Survived")

rf.pred <- predict(rf.model, newdata = test1, type = "response")
rf.pred <- as.numeric(rf.pred == 1)
rf.submission <- test$PassengerId
rf.submission <- as.data.frame(cbind(rf.submission, rf.pred))
names(rf.submission) <- c("PassengerId", "Survived")


write.csv(submission, file = "titanicsubmission2.csv", row.names = F)
write.csv(logistic.submission, file = "logistictitanic.csv", row.names = F)
write.csv(rf.submission, file = "rftitanic", row.names = F)
