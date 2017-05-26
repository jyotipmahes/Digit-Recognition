#Convulational network using R

#Set Current directory to file location
setwd("C:/Users/Jyoti Prakash/Desktop/Digit Recognition/")

#Using mxnet for the CNN network
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("mxnet")
library(mxnet)


# Load train and test datasets
train1=read.csv("train.csv")
ftest=read.csv("test.csv")

# Set up train and test datasets
a=sample(nrow(train1),30000)
train=train1[a,]
train =data.matrix(train)
train_x = t(train[, -1])
train_y = train[, 1]
train_array = train_x
dim(train_array) = c(28, 28, 1, ncol(train_x))

test_x = t(train1[-a, -1])
test_y <- t(train1[-a, 1])
test_array = data.matrix(test_x)
dim(test_array) = c(28, 28, 1, ncol(test_x))


# Set up the symbolic model

data = mx.symbol.Variable('data')
# 1st convolutional layer
conv_1 = mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1 = mx.symbol.Activation(data = conv_1, act_type = "tanh")
pool_1 = mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
# 1st fully connected layer
flatten <- mx.symbol.Flatten(data = pool_2)
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)

# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Device used. CPU in my case.
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(NN_model,
                                     X = train_array,
                                     y = train_y,
                                     ctx = devices,
                                     num.round = 30,
                                     array.batch.size = 40,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Testing
#-------------------------------------------------------------------------------

# Predict labels
predicted = predict(model, test_array)
# Assign labels
predicted_labels = max.col(t(predicted)) - 1
sum(diag(table(predicted_labels,test_y)))/ncol(test_y)

test_x = t(ftest)
ftest_array = data.matrix(test_x)
dim(ftest_array) = c(28, 28, 1, ncol(test_x))

pred1=predict(model,ftest_array)
labels = max.col(t(pred1)) - 1
ImageId=c(1:nrow(ftest))
out=as.data.frame(ImageId)
out$Label=labels
write.csv(out,"Outv1.csv",row.names=FALSE)



################################################################################
#                           OUTPUT
################################################################################
#
# 0.975
#