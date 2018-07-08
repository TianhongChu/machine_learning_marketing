library(tidyverse)
setwd("/Users/Rainbow/Desktop/machine_learning_final2")
# read in the transaction file
ord = read.csv("orders.csv")
dim(ord)
head(ord)

# the date of the offer was 11/25/2014, so t is time since action
library(lubridate)
ord$t = as.numeric(ymd("2014/11/25") - dmy(ord$orddate))/365.25
summary(ord$t)
hist(ord$t)

# read in the customer file with one row per customer
customer = read.csv("customer.csv")
names(customer)
head(customer)
table(customer$train)

# rollup to customer id - year level 
# create features - yearly monetary value
library(tidyr)
library(lubridate)
ord$datenew<-dmy(ord$orddate)
yrs <- sort(unique(year(ord$datenew)))
ltv_y = ord %>%
  dplyr::mutate(yr = year(datenew)) %>%
  dplyr::group_by(id, yr) %>%
  dplyr::summarise(m_year = sum(price*qty)) %>%
  tidyr::spread(yr, m_year, fill = 0) %>%
  setNames(c("id", paste("year", yrs, sep="")))
head(ltv_y)

# rollup order file to create RFM variables
# create features - tof(time on file), recency, number of items bought, frequency(number of orders), monetary value, breadths of category,
# average monetary value per order, number of items per order, average monetary value per year,average items bought per year, purchase cycle
rfm = ord %>%
  dplyr::group_by(id) %>%
  dplyr::summarise(tof=max(t), r = min(t), fitem=n(), ford=n_distinct(ordnum), m=sum(price*qty), fcat = n_distinct(category)) %>%
  dplyr::mutate(avg_m_f = m/ford,  avg_item_f =fitem/ford, avg_m_y = m/tof, avg_f_y=ford/tof,avg_item_y = fitem/tof, pcycle = (tof-r)/(ford-1))
head(rfm)
summary(rfm)
dim(rfm)
rfm$pcycle[is.na(rfm$pcycle)] <- mean(rfm$pcycle, na.rm = T) # mean fill

# create active24, active12, active4 dummy variable indicating whether a customer has bought in 
# the past 24 months, 12 months, 4 months
rfm$active24=ifelse(rfm$r<=2,1,0) 
rfm$active12=ifelse(rfm$r<=1,1,0) 
rfm$active4=ifelse(rfm$r<=0.3,1,0) 

# create old and recent dummy variable 
# old_r <- 1 if a customer has been on file for more than a year and has bought in less than a year
rfm$old_r <- 0
rfm$old_r[rfm$tof > 1 & rfm$r < 1] <- 1

# roll up 24 month RFM features - number of items bought, frequency, monetary value
rfm24 <- rfm[rfm$active24==1,] %>%
  dplyr::left_join(ord[ord$t<=2,], by = "id") %>%
  dplyr::group_by(id) %>%
  dplyr::summarise(fitem24 = n(), ford24=n_distinct(ordnum),m24=sum(price*qty)) 

# roll up 12 month RFM features - number of items bought, frequency, monetary value
rfm12 <- rfm[rfm$active12==1,] %>%
  dplyr::left_join(ord[ord$t<=1,], by = "id") %>%
  dplyr::group_by(id) %>%
  dplyr::summarise(fitem12 = n(), ford12=n_distinct(ordnum),m12=sum(price*qty))

# roll up 4 months RFM features - number of items bought, frequency, monetary value
rfm4 <- rfm[rfm$active4==1,] %>%
  dplyr::left_join(ord[ord$t<=0.3,], by = "id") %>%
  dplyr::group_by(id) %>%
  dplyr::summarise(fitem4 = n(), ford4=n_distinct(ordnum),m4=sum(price*qty))

# this shows you how you can roll up order file counting purchases by category
cats = sort(unique(ord$category))  # list of all unique categories
cats
rfm2 = ord %>%
  dplyr::group_by(id, category) %>%
  dplyr::summarise(f= n()) %>%
  spread(category,f, fill=0)  %>%
  setNames(c("id", paste("f", cats, sep="")))
head(rfm2)
summary(rfm2)

# this joins the customer file, RFM, yearly monetary value, RFM24,RFM12, RFM4 and purchase_by_category tables
all <- customer %>%
  dplyr::left_join(rfm, by = "id") %>%
  dplyr::left_join(ltv_y, by = "id") %>%
  dplyr::left_join(rfm24, by = "id") %>%
  dplyr::left_join(rfm12, by = "id") %>%
  dplyr::left_join(rfm4, by = "id") %>%
  dplyr::left_join(rfm2, by = "id")
summary(all)
names(all)

# replace NAs with 0 in joined data frame
colnames(all)
all[,c(28:36)][is.na(all[,c(28:36)])] <- 0

# roll up seasonality data 
# create features - monthly frequency(number of orders)
library(lubridate)

months <- sort(unique(month(ord$datenew)))
season <- ord %>%
  dplyr::mutate(ord_month = month(datenew)) %>%
  dplyr::group_by(id, ord_month) %>%
  dplyr::summarise(ford_month = n_distinct(ordnum)) %>%
  spread(ord_month,ford_month, fill=0)  %>%
  setNames(c("id", paste("month", months, sep=""))) 
str(season)

season1 <- season %>%
  dplyr::group_by(id) %>%
  dplyr::mutate(christmas = month11 + month12) %>%
  dplyr::select(id, christmas)

colnames(season1) 

# add seasonality data to the dataframe
all <- all %>%
  dplyr::left_join(season1, by = "id")
colnames(all)

# Note that the dependent variable is in col 3 and the predictors are in columns 4-68 for variale selection

# This command logs all of the predictor variables except categorical variables
all$active12 <- factor(all$active12)
all$active4 <- factor(all$active4)
all$old_r <- factor(all$old_r)
for(i in 4:15) all[[i]] = log(all[[i]]+1)
for(i in 20:67) all[[i]] = log(all[[i]]+1)
summary(all)
names(all)

train = (all$train==1)  # create logical train variable
table(train)
all_train <- all[train==1,-c(20:24,66)]
all_test <- all[train==0,-c(20:24,66)]
names(all_train)


## ridge regression
library(caret)
library(glmnet)

# cross validation
set.seed(12345)
fit_ridge <- cv.glmnet(data.matrix(all_train[ , 4:61]), all_train$logtarg, alpha = 0, nfolds = 10, lambda = seq(0,1,0.001))
fit_ridge$lambda.min
coef(fit_ridge,s="lambda.min")

pred_ridge_direct <- predict(fit_ridge, newx = data.matrix(all_train[ , 4:61]), s = "lambda.min")
plot(pred_ridge_direct, all_train$logtarg)
abline(a = 0, b = 1)

pred_ridge_direct[pred_ridge_direct < 0] <- 0 # change prediction values to 0 if values are less than 0 
sqrt(mean((pred_ridge_direct  -  all_train$logtarg)^2))

# predict logtarg on test dataset
predict_ridge_test<-predict(fit_ridge, newx = data.matrix(all_test[ , 4:61]), s = "lambda.min")
predict_ridge_test[predict_ridge_test < 0] <- 0 # change prediction values to 0 if values are less than 0 

# output file
try0606_4 <- cbind(all_test$id, predict_ridge_test)
colnames(try0606_4) <- c("id", "logtarg")
write.csv(try0606_4, "try0606_4.csv", row.names=F)

