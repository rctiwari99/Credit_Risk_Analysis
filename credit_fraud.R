credit_data <- read.csv("C:/Users/rc_as/Desktop/Data_science/GitHUb/credit_risk-modelling-master/credit_data.csv", header = TRUE)
str(credit_data)
table(credit_data$grade)
table(credit_data$home_ownership)
summary(credit_data)
# cross table
library(gmodels)
CrossTable(credit_data$loan_status)
CrossTable(credit_data$grade, credit_data$loan_status, prop.r = TRUE, prop.c= FALSE, prop.t = FALSE, prop.chisq = FALSE)
# The proportion of defaults increases when the credit rating moves from A to G.
library(ggplot2)
library(dplyr)
# EDA
credit_data %>% ggplot(aes(annual_inc)) + geom_histogram()+ ggtitle("Histogram of Annual Income")+ xlab("Annual Income")
credit_data %>% ggplot(aes(x = factor(loan_status), y = annual_inc)) +geom_boxplot()
credit_data %>% ggplot(aes(x = factor(loan_status), y = loan_amnt)) +geom_boxplot()
credit_data %>% ggplot(aes(loan_amnt)) + geom_histogram(binwidth = 1000)+ ggtitle("Histogram of Loan Amount")+xlab("Loan Amount")

CrossTable(credit_data$loan_status, credit_data$home_ownership, prop.r = TRUE, prop.c= TRUE, prop.t = FALSE, prop.chisq = FALSE)
credit_data %>%ggplot(aes(home_ownership))+geom_bar()+facet_wrap(~loan_status)
credit_data %>%ggplot(aes(y = annual_inc, x = 1)) + geom_boxplot()
# wrong data in the data set identification
plot(credit_data$annual_inc)
plot(credit_data$age)
credit_data %>%ggplot(aes(age, annual_inc))+geom_point()
index_wrongdata <-which(credit_data$age > 110)
credit_data[index_wrongdata, ]
# removing wrong data
credit_data <- credit_data[-index_wrongdata, ]
summary(credit_data)
# not removing outliers from annual_inc and loan_amnt, because i want to work on all type of observations dealing with missing values
# deleting observations / imputing median / coarse classification
# here i am using coarse classification by using different bins # bins can be taken from quantile to get an equal distribution
quantile(credit_data$int_rate, na.rm = TRUE)
# Making the necessary replacements in the coarse classification  
credit_data$ir_cat<- rep(NA, length(credit_data$int_rate))
credit_data$ir_cat[which(credit_data$int_rate <= 8)] <- "0-8"
credit_data$ir_cat[which(credit_data$int_rate > 8 & credit_data$int_rate <= 11)] <- "8-11"
credit_data$ir_cat[which(credit_data$int_rate > 11 & credit_data$int_rate <= 13.5)] <- "11-13.5"
credit_data$ir_cat[which(credit_data$int_rate > 13.5)] <- "13.5+"
credit_data$ir_cat[which(is.na(credit_data$int_rate))] <- "Missing"
credit_data$ir_cat <- as.factor(credit_data$ir_cat)
# Look at your new variable using plot()
plot(credit_data$ir_cat)
# same coarse classification is repeated for employment length using suitable bins
# bins can be taken from quantile to get an equal distribution
quantile(credit_data$emp_length, na.rm = TRUE)
credit_data$el_cat<- rep(NA, length(credit_data$emp_length))
credit_data$el_cat[which(credit_data$emp_length <= 2)] <- "0-2"
credit_data$el_cat[which(credit_data$emp_length > 2 & credit_data$emp_length <= 4)] <- "3-4"
credit_data$el_cat[which(credit_data$emp_length > 4 & credit_data$emp_length <= 8)] <- "5-8"
credit_data$el_cat[which(credit_data$emp_length > 8)] <- "8+"
credit_data$el_cat[which(is.na(credit_data$emp_length))] <- "Missing"
credit_data$el_cat <- as.factor(credit_data$el_cat)
# Look at your new variable using plot()
plot(credit_data$el_cat)
# deleting original columns for emp_length and int_rate
credit_data_2 <- credit_data 
credit_data_2$int_rate <- NULL 
credit_data_2$emp_length <- NULL
credit_data_2$X <- NULL
credit_clean <- credit_data_2
# splitting data into training and validation
set.seed(567)
index <- sample(1:nrow(credit_clean), 2/3*nrow(credit_clean))
credit_training <- credit_clean[index,]
credit_testing <- credit_clean[-index,]
# logistic regression
summary(credit_training)
str(credit_training)
logit_model <- glm(loan_status ~. ,data = credit_training, family = "binomial")
summary(logit_model)
predictions <- predict(logit_model, newdata = credit_testing, type = "response")
# checking range of predicted probabilities to check wheather the model distinguishes good and bad customers properly
range(predictions)
# checking area under curve of model
library(pROC)
auc(credit_testing$loan_status, predictions)
# selecting a proper threshold / cut-off value # pred_15 <- ifelse(predictions > 0.15, 1, 0)
# table(credit_testing$loan_status, pred_15) # pred_20 <- ifelse(predictions > 0.20, 1, 0)
# table(credit_testing$loan_status, pred_20)
# Forward stepwise approach to add predictors to the model one-by-one until no additional benefit is seen
# Null model with no predictors
null_model <- glm(loan_status ~ 1, data =credit_training, family = binomial(link = cloglog))
# Full model using all predictors
full_model <- glm(loan_status ~. , data = credit_training, family = binomial(link = cloglog))
# forward stepwise algorithm to build a parsimonious model
step_model <- step(null_model, scope = list(lower = null_model, upper = full_model), direction = "forward")
summary(step_model)
prediction_stepmodel <- predict(step_model, newdata = credit_testing, type = "response")
quantile(prediction_stepmodel)
auc(credit_testing$loan_status, prediction_stepmodel)
# Computing bad rate (percentage of defaults) for a given acceptance rate  by the bank
# suppose the acceptance rate given is 80% of applicants
# computing cutoff based on acceptance rate
cuttoff <- quantile(prediction_stepmodel, 0.8) # 80% of max to min data range
# based on cuttoff obtaining binary predictions
bin_pred <- ifelse(prediction_stepmodel > cuttoff, 1, 0)
# Obtaining the actual default status for the accepted loans
index_accp <- which(bin_pred == 0)
accp_customers <- credit_testing$loan_status[index_accp]
# Obtain the bad rate for the accepted loans
bad_rate <- sum(accp_customers == 1) / length(accp_customers)
# The strategy table and strategy curve
# Repeating the calculations from the previous step for several acceptance rates, a strategy table is obtained
# This table can be used to better define a acceptance stategy
# creating a custum function to compute cuttoff, bad rate for a acceptance rate that are multiples of 5%
strategy_bank <- function(prob_of_def){
  cutoff=rep(NA, 21)
  bad_rate=rep(NA, 21)
  accept_rate=seq(1,0,by=-0.05)
  for (i in 1:21){
    cutoff[i]=quantile(prob_of_def,accept_rate[i])
    pred_i=ifelse(prob_of_def> cutoff[i], 1, 0)
    pred_as_good=credit_testing$loan_status[pred_i==0]
    bad_rate[i]=sum(pred_as_good == 1)/length(pred_as_good)}
  table=cbind(accept_rate,cutoff=round(cutoff,4),bad_rate=round(bad_rate,4))
  return(list(table=table,bad_rate=bad_rate, accept_rate=accept_rate, cutoff=cutoff))
}
# strategy table
strategy_table <- strategy_bank(prediction_stepmodel)

# strategy curve
strategy_curve <- plot(strategy_table$accept_rate, strategy_table$bad_rate, 
                       type = "l", xlab = "Acceptance rate", ylab = "Bad rate", 
                       lwd = 2, main = "logistic regression")