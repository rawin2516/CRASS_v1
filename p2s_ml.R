####################################
# CRASS MACHINE LEARNING ALGORITHM #
####################################


#######################
# Author: Awin Salian #
#######################


#Last Updated: 2018/01/11#



#Set Random Seed
set.seed(2)


#############################################################################################################
#                                                                                                           #
#                                        LOAD REQUIRED PACKAGES                                             #
#                                                                                                           #
#############################################################################################################

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('Boruta') # Var Importance Check

library('mice') # imputation

library('randomForest') # classification algorithm

library('caret') # ml algorithm

library('cluster') # clustering

library('caretEnsemble') # Ensemble modeling

library('pROC') # AUC - ROC

library('doParallel') # parallel processing

library('DMwR') # smore

library('ROSE')

#set_working directory

setwd('C:/Users/p2s/Documents/ML')
#############################################################################################################
#                                                                                                           #
#                                         READING DATA                                                      #
#                                                                                                           #
#############################################################################################################

cs <- read.csv("train_ml_2.csv") #reading from a csv file, this will be changed to a MySQL database


#######################################Train and test######################################


index <- sample(nrow(cs), floor(0.85 * nrow(cs)))

cs$Verdict <- as.factor(cs$Verdict)

train_cs <- cs[index, -c(1, 2)]

test_cs <- cs[-index, -c(1, 2)]


###########################Search for NAs/ preprocessing#####################################################  

cs.miss.model <- preProcess(train_cs, method = c("center", "scale", "YeoJohnson", "knnImpute")) # Pre processing the data with centering, scaling, transforming each variab and imputing missing values

train_cs_t <- predict(cs.miss.model, train_cs) #Applying the preprocessing model on the training set   

test_cs_t <- predict(cs.miss.model, test_cs)  #Applying the preprocessing model on the test set 

#############################################Sampling####################################

##########check if sampling is required###################


classification_balance <- as.data.frame(table(train_cs_t$Verdict)) #check the number of 1s and 0s in the dataset

num_0 <- classification_balance$Freq[classification_balance$Var1 == 0] 
num_1 <- classification_balance$Freq[classification_balance$Var1 == 1]

classification_balance_percentages <- c(num_0/(num_0 + num_1) , num_1/(num_0 + num_1)) #percentage of each classification bin in the dataset

#Resampling based on the criteria if one bin outweighs the other by 10%

if(abs(classification_balance_percentages[1] - classification_balance_percentages[2]) >= 0.1){
  
  train_cs_t <-
    ovun.sample(
      Verdict ~ .,
      data = train_cs_t,
      method = "both",
      p = 0.5,
      N = 500,
      seed = 2
    )$data
  
}

#############################################################################################################
#                                                                                                           #
#                                         BUILD THE MODEL                                                   #
#                                                                                                           #
#############################################################################################################

#############################################CV and train control###########################################



myControl <- trainControl(
  method = "repeatedcv", #Repeated Cross-Validation
  
  number = 10, # 10 folds
  
  repeats = 1, # No repeats
  
  verboseIter = TRUE,
  
  returnResamp = "all"
  
)


############Train RF##################################

tunegrid <- expand.grid(.mtry=c(sqrt(ncol(x))))
modellist <- list()
for (ntree in c(1000, 1500, 2000, 2500)) {
  set.seed(seed)
  model_output_rf <- train(as.factor(Verdict) ~ ., data = train_cs_t,
                                  trControl = myControl, metric = "ROC",
                                  method = "rf")  
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)#####choose the best by EDA######



########Train GLM###########


model_output_glm <- train(as.factor(Verdict) ~ ., data = train_cs_t,
                          trControl = myControl, metric = "ROC",
                          method = "glm")

##############Prediction on test data#############

pred_rf <- predict(model_output_rf, newdata = test_cs_t, type = 'prob')

pred_glm <- predict(model_output_rf, newdata = test_cs_t, type = 'prob')

pred_average <- rowMeans(cbind(pred_rf$`1`, pred_glm$`1`))


##################Convert the three columns into 1,0 based on cutoff 0.4###############

pred_rf_raw <- vector("numeric", length(pred_rf))

pred_glm_raw <- vector("numeric", length(pred_rf))

pred_average_raw <- vector("numeric", length(pred_rf))


for(i in 1:length(pred_rf_raw)){
  
  if(pred_rf$`1`[i]>=0.4){
    
    pred_rf_raw[i] <- 1
    
  }else{
    
    pred_rf_raw[i] <- 0
  }
  
  if(pred_glm$`1`[i]>=0.4){
    
    pred_glm_raw[i] <- 1
    
  }else{
    
    pred_glm_raw[i] <- 0
  }
  
  if(pred_average$`1`[i]>=0.4){
    
    pred_average_raw[i] <- 1
    
  }else{
    
    pred_average_raw[i] <- 0
  }
  
  
}


############Check for AUC - ROC#############

area_roc_rf <- as.numeric(roc(test_cs_t$Verdict, pred_rf_raw)$auc)

area_roc_glm <- as.numeric(roc(test_cs_t$Verdict, pred_glm_raw)$auc)

area_roc_average <- as.numeric(roc(test_cs_t$Verdict, pred_average_raw)$auc)


############Choose best model by ROC##############

###########Run the best model on entire data set#####


##########taking the example that a glm model turned out to be the best########


final_pred <- predict(model_output_glm, newdata=cs, type = 'prob')

########Run a linear regression with the scores as the output#######

cs$score <- final_pred$`1`

reg_score <- lm(score ~ ., data= cs)

######Read coefficients from the reg_score table####

