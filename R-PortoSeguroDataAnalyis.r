
install.packages("tidyverse")
install.packages("caret")
install.packages("verification")
install.packages("repr")
install.packages("arm")
install.packages("cowplot")
install.packages("LogicReg")
install.packages("superml")

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(verification))
library(arm)
library(cowplot)
library(LogicReg)
library(superml)

df <- read.csv("train.csv")
str(df)

df_test <- read.csv("test.csv")

summary(df)

# set seed for reproducibility
set.seed(123)

# making a train index
train_index <- sample(c(TRUE, FALSE), replace = TRUE, size = nrow(df), prob = c(0.2, 0.8))

# split the data according to the train index
training <- as.data.frame(df[train_index, ])
testing <- as.data.frame(df[!train_index, ])

logmod <- glm(target ~ . - id, data = training, family = binomial(link = 'logit'))

summary(logmod)

preds <- predict(logmod, newdata = testing, type = "response")

y_pred_num <- ifelse(preds > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <- testing$target

# Accuracy
mean(y_pred == y_act)  

coefplot(logmod, vertical=FALSE, mar=c(5.5,2.5,2,2))

tapply(preds, testing$target, mean)

# Confusion matrix for threshold of 0.1
table(testing$target, preds > 0.1)

# Confusion matrix for threshold of 0.3
table(testing$target, preds > 0.3)

# Confusion matrix for threshold of 0.5
table(testing$target, preds > 0.5)

# Confusion matrix for threshold of 0.7
table(testing$target, preds > 0.7)

# Use 'scale' to normalize
table(testing$target, preds > 0.1)

data <- as.matrix(table(testing$target, preds > 0.1))
heatmap(data, scale="column")

install.packages("ROCR")
library(ROCR)

ROCRpred = prediction(preds, testing$target)

ROCRperf = performance(ROCRpred, "tpr", "fpr")

## AUC Value

auc.tmp <- performance(ROCRpred,"auc");
auc <- as.numeric(auc.tmp@y.values)

auc

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

install.packages("data.table")

library(data.table)

train_l <- as.tibble(fread('train.csv', na.strings=c("-1","-1.0")))

install.packages("corrplot")

library("corrplot")

training %>%
  select(-starts_with("ps_calc"), -ps_ind_10_bin, -ps_ind_11_bin, -ps_car_10_cat, -id) %>%
  mutate_at(vars(ends_with("cat")), funs(as.integer)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.integer)) %>%
  mutate(target = as.integer(target)) %>%
  cor(use="complete.obs", method = "spearman") %>%
  corrplot(type="lower", tl.col = "black",  diag=FALSE)

corr_check <- function(Dataset, threshold){
  matriz_cor <- cor(Dataset)
  matriz_cor

  for (i in 1:nrow(matriz_cor)){
    correlations <-  which((abs(matriz_cor[i,i:ncol(matriz_cor)]) > threshold) & (matriz_cor[i,i:ncol(matriz_cor)] != 1))
  
    if(length(correlations)> 0){
      lapply(correlations,FUN =  function(x) (cat(paste(colnames(Dataset)[i], "with",colnames(Dataset)[x]), "\n")))
     
    }
  }
}

corr_check(training, 0.50)

print("training$ps_ind_11_bin")
sum(training$ps_ind_11_bin==-1)
print("training$ps_ind_02_cat")
sum(training$ps_ind_02_cat==-1)
par(mfrow=c(1,2)) 
hist(training$ps_ind_11_bin)
hist(training$ps_ind_02_cat)

training$ps_ind_11_bin <- NULL 

print("training$ps_ind_12_bin")
sum(training$ps_ind_12_bin==-1)
print("training$ps_ind_01")
sum(training$ps_ind_01==-1)
par(mfrow=c(1,2)) 
hist(training$ps_ind_12_bin)
hist(training$ps_ind_01)

training$ps_ind_12_bin <- NULL 

print("training$ps_reg_01")
sum(training$ps_reg_01==-1)
print("training$ps_ind_02_cat")
sum(training$ps_ind_01==-1)
par(mfrow=c(1,2)) 
hist(training$ps_reg_01)
hist(training$ps_ind_01)

print("training$ps_car_04_cat")
sum(training$ps_car_04_cat==-1)
print("training$ps_ind_08_bin")
sum(training$ps_ind_08_bin==-1)
par(mfrow=c(1,2)) 
hist(training$ps_car_04_cat)
hist(training$ps_ind_08_bin)

training$ps_ind_08_bin <- NULL 

print("training$ps_car_04_cat")
sum(training$ps_car_04_cat==-1)
print("training$ps_ind_09_bin")
sum(training$ps_ind_09_bin==-1)
par(mfrow=c(1,2)) 
hist(training$ps_car_04_cat)
hist(training$ps_ind_09_bin)

training$ps_ind_09_bin <- NULL 

print("training$ps_car_13")
sum(training$ps_car_13==-1)
print("training$ps_ind_01")
sum(training$ps_ind_01==-1)
par(mfrow=c(1,2)) 
hist(training$ps_car_13)
hist(training$ps_ind_01)

rf <- RFTrainer$new()
gst <-GridSearchCV$new(trainer = rf,
parameters = list(n_estimators = c(100),
max_depth = c(5,2,10)),
n_folds = 3,
scoring = c('accuracy','auc'))

rf <- RFTrainer$new()
gst <-GridSearchCV$new(trainer = rf,
parameters = list(n_estimators = c(100),
max_depth = c(5,2,10)),
n_folds = 3,
scoring = c('accuracy','auc'))
data("train_l")
gst$fit(train_l, "target")

gst$accuracy

gst$auc

testing$ps_ind_11_bin <- NULL 
testing$ps_ind_12_bin <- NULL 
testing$ps_ind_08_bin <- NULL 
testing$ps_ind_09_bin <- NULL 

logmod_second <- glm(target ~ . - id, data = training, family = binomial(link = 'logit'))

preds <- predict(logmod_second, newdata = testing, type = "response")

y_pred_num <- ifelse(preds > 0.2, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <- testing$target

# Accuracy
mean(y_pred == y_act)  

ROCRpred_second = prediction(preds, testing$target)

ROCRperf = performance(ROCRpred_second, "tpr", "fpr")

plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

auc.tmp <- performance(ROCRpred_second,"auc");
auc <- as.numeric(auc.tmp@y.values)

auc

# Use 'scale' to normalize
table(testing$target, preds > 0.1)

data <- as.matrix(table(testing$target, preds > 0.1))
heatmap(data, scale="column")


