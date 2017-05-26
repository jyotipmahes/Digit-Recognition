#Digit Recognition - Kaggle competition
#Jyoti Prakash

#Set Current directory to file location
setwd("C:/Users/Jyoti Prakash/Desktop/Digit Recognition/")


#Loading data files
train=read.csv("train.csv")
test=read.csv("test.csv")

#Analyzing data 
str(train)
summary(train)

#Data Preprocessing
#removing label variable
wl.train=train[-1]

#Removing nearzero variables
nearz=nearZeroVar(wl.train)
postnz.train=wl.train[,-nearz]

#testset
postnz.test=test[,-nearz]

#PCA
prepca=preProcess(postnz.train,method=c("pca"),thresh=.95)
proTrain=predict(prepca,postnz.train)
ftrain=data.frame(label=train$label,proTrain)

ftest=predict(prepca,postnz.test)

#Building a random forest model
library(randomForest)
set.seed(123)
n=25000
row=sample(1:nrow(ftrain),n)
ftrain[,1]=as.factor(ftrain[,1])
tdata=ftrain[row,]

#Random forest model
model=randomForest(label~.,data=tdata)
testd=ftrain[-row,]
pred=predict(model,newdata=testd,type="class")
table(testd$label,pred)
sum(diag(table(testd$label,pred)))/nrow(testd)

#Neural network
#feats =names(testd[-1])

# Concatenate strings
#f = paste(feats,collapse=' + ')
#f =paste('label ~',f)

# Convert to formula
#f = as.formula(f)

#install.packages("neuralnet")
#library(neuralnet)
#nn = neuralnet(f,tdata,hidden=c(10,10),linear.output=FALSE)
#pred=as.data.frame(compute(nn,tdata[-1]))
#pred$net.result=sapply(pred$net.result,round,digits=0)
#table(tdata$label,pred$net.result)


#Multinomial classification
#install.packages("nnet")
library(nnet)
library(caret)
library(RCurl)
library(Metrics)

set.seed(12)
#mmodel=multinom(label~.,data=tdata,maxit=1500,trace=T,MaxNWts=7860)
#pred=predict(mmodel,type="class",newdata=testd)
#postResample(testd$label,pred)
#very poor acurracy 45%

library(neuralnet)


colname = names(train)
strpred = paste(colname[!colname %in% "label"], collapse = " + ")

alllevel = levels(as.factor(train$label))
for(tlabel in unique(alllevel)) {
	train[paste("lbl", tlabel, sep = "_")] <- ifelse(train$label == tlabel, 1, 0)
}
colname=names(train)
stry = paste(colname[substr(colname,1,4) %in% "lbl_"], collapse = " + ")
nnformula = as.formula(paste(stry, " ~ ", strpred))
n=1000
row=sample(1:nrow(train),n)
tdata=train[row,]
testdata=train[-row,]
digit.nn = neuralnet(nnformula, data=tdata, hidden =10, linear.output=FALSE)

pr.nn = compute(digit.nn, testdata[,2:785])
testdata$pred.nn = max.col(pr.nn$net.result[,1:10])
testdata$pred.nn = testdata$pred.nn -1
testdata$pred.nn <- as.factor(testdata$pred.nn)
table(testdata$label, testdata$pred.nn)
tmp_acc =table(testdata$pred.nn, testdata$label)
sum(diag(tmp_acc))/nrow(testdata)
#46.81 % for neuralnet

# this gives 0.89 accuracy with threshold 5 and 10 min run. Try with more threshold















#Predicting output
pred1=predict(model,newdata=test,type="class")
ImageId=c(1:nrow(test))
out=as.data.frame(ImageId)
out$Label=pred1
write.csv(out,"Outv1.csv",row.names=FALSE)



