title: "Classification summative assignment"
author: bvck63

hotel<- read.csv("https://www.louisaslett.com/Courses/MISCADA/hotels.csv", header=TRUE)

#data summary
install.packages("skimr")
library("skimr")
skim(hotel)
head(hotel)

#simple visualization of the data
install.packages("DataExplorer")
library("DataExplorer")
install.packages("tidyverse")
library("tidyverse")
DataExplorer::plot_bar(hotel, ncol = 2)
DataExplorer::plot_histogram(hotel, ncol = 2)
#Use target variable "is_canceled" to split out continues variables
DataExplorer::plot_boxplot(hotel, by = "is_canceled", ncol = 2)
#Explore relationship between different values
hotel.bycountry <- hotel %>% 
  group_by(country) %>% 
  summarise(total = n(),
            cancellations = sum(is_canceled),
            pct.cancelled = cancellations/total*100)
install.packages("rnaturalearth")
library("rnaturalearth")
install.packages("rnaturalearthdata")
library("rnaturalearthdata")
install.packages("rgeos")
library("rgeos")
world <- ne_countries(scale = "medium", returnclass = "sf")
world2 <- world %>%
  left_join(hotel.bycountry,
            by = c("iso_a3" = "country"))
# plot the map with ggplot and set the fill colour of each country to be the percentage of customers who cancelled.
ggplot(world2) +
  geom_sf(aes(fill = pct.cancelled))
hotel <- hotel %>%
  select(-reservation_status, -reservation_status_date) %>% 
  mutate(kids = case_when(
    children + babies > 0 ~ "kids",
    TRUE ~ "none"
  ))
hotel.par <- hotel %>%
  select(hotel, is_canceled, kids, meal, customer_type) %>%
  group_by(hotel, is_canceled, kids, meal, customer_type) %>%
  summarize(value = n())
install.packages("ggforce")
library("ggforce")
#visualise the relationship between levels and the cancellation status
ggplot(hotel.par %>% gather_set_data(x = c(1, 3:5)),
       aes(x = x, id = id, split = y, value = value)) +
  geom_parallel_sets(aes(fill = as.factor(is_canceled)),
                     axis.width = 0.1,
                     alpha = 0.66) + 
  geom_parallel_sets_axes(axis.width = 0.15, fill = "lightgrey") + 
  geom_parallel_sets_labels(angle = 0) +
  coord_flip()

#remove some variables
hotel <- hotel %>% 
  select(-country, -reserved_room_type, -assigned_room_type, -agent, -company,
         -stays_in_weekend_nights, -stays_in_week_nights,-hotel,-arrival_date_month,-meal,-market_segment,-distribution_channel,-deposit_type,-customer_type)
hotel <- hotel %>% 
  select(-babies, -children,-kids)

#Model fitting
#Task definition
install.packages("mlr3verse")
library("mlr3verse")
hotel$is_canceled<-as.factor(hotel$is_canceled)
task <- TaskClassif$new(id = "hotel",
                        backend = hotel,
                        target = "is_canceled",
                        positive = "1")

#Learner definition
install.packages("mlr3learners")
library("mlr3learners")
#initailize logic regression
learner_logreg=lrn("classif.log_reg")
learner_logreg

#Data training(train dataset 80%, test dataset 20%)
train=sample(task$row_ids,0.8*task$nrow)
test=setdiff(task$row_ids,train)
head(train)
learner_logreg$train(task,row_ids =train )
learner_logreg$model
class(learner_logreg$model)
summary(learner_logreg$model)

#Random forest
install.packages("precrec")
library(precrec)
learner_rf=lrn("classif.ranger",importance="permutation",predict_type="prob")
learner_rf$train(task,row_ids=train)
pred_rf=learner_rf$predict(task,row_ids=test)
pred_rf$confusion
autoplot(pred_rf,type='roc',main='Random forest')

#CART 
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_cart$train(task, row_ids = train)
pred_cart<-lrn_cart$predict(task, row_ids = test)
autoplot(pred_cart,type='roc',main='CART')

#plot code
install.packages("caret")
library(caret)
pred_cart$confusion
pred_rf$confusion
pred_logreg$confusion
ctable <- as.table(matrix(c(5389, 2491, 3482, 12516), nrow = 2, byrow = TRUE))
fourfoldplot(ctable, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Cart prediction")
ctable2 <- as.table(matrix(c(5880, 1351, 2991, 13656), nrow = 2, byrow = TRUE))
fourfoldplot(ctable2, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Random forest prediction")
ctable3 <- as.table(matrix(c(4382, 1849,  4489, 13158), nrow = 2, byrow = TRUE))
fourfoldplot(ctable3, color = c("cyan", "pink"),
             conf.level = 0, margin = 1, main = "Logistic Regression prediction")

#Logistic Regression
pred_logreg=learner_logreg$predict(task,row_ids = test)
pred_logreg
pred_logreg$confusion
learner_logreg$predict_type="prob"
learner_logreg$predict(task,row_ids = test)
autoplot(pred_logreg,type='roc',main='Logistic Regression')

#Performance evaluation
#cross-validation
resampling=rsmp("cv",folds=5)
rr=resample(task,learner = learner_logreg,resampling = resampling)
rf=resample(task,learner = learner_rf,resampling = resampling)
cart=resample(task,learner = lrn_cart,resampling = resampling)
rr$aggregate()
measures=msrs(c("classif.fnr","classif.fpr"))
rr$aggregate(measures) 

rf$aggregate()
measures_rf=msrs(c("classif.fnr","classif.fpr"))
rf$aggregate(measures_rf) 

cart$aggregate()
measures_cart=msrs(c("classif.fnr","classif.fpr"))
cart$aggregate(measures_cart) 

#Performance Comparision
install.packages("data.table")
library("data.table")
install.packages("xgboost")
library("xgboost")
set.seed(212) # set seed for reproducibility


# Define task
hotel_task <- TaskClassif$new(id = "hotel",
                              backend = hotel,
                              target = "is_canceled",
                              positive = "1")

# Cross validation resampling strategy
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(hotel_task)

# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

# Define a super learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Factors coding pipeline
pl_factor <- po("encode")

# Now define the full pipeline
spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop") # This passes through the original features adjusted for
      # missingness to the super learner
    )),
  # Last group needing factor encoding
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

res_spr <- resample(hotel_task, spr_lrn, cv5, store_models = TRUE)

res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))

#Automating the tuning
install.packages("mlr3tuning")
library(mlr3tuning)
# learner_lr$param_set
learner=lrn("classif.ranger")
learner$param_set
search_space=ps(
  num.trees=p_int(lower=500,upper=1000),
  mtry=p_int(lower=1,upper = 13),
  min.node.size=p_int(lower = 2,upper=10)
)
terminator = trm("evals", n_evals = 10)
tuner = tnr("random_search")
at = AutoTuner$new(
  learner = learner,
  resampling = rsmp("holdout"),
  measure = msr("classif.ce"),
  search_space = search_space,
  terminator = terminator,
  tuner = tuner
)
at

at$train(hotel_task,row_ids = train)
at$predict(hotel_task,row_ids = test)

#'ce' plot code
rf$score()
x=c(1,2,3,4,5)
y=c(0.1926041,0.1845632,0.1921853,0.1848982,0.1892537)
plot(x,y,xlab="iteration",ylab="error of classification",type="o")
x1=c(1,2,3,4,5,6,7,8,9,10)
y1=c(0.1645256,0.1656249,0.1612275,0.16418,0.2181424,0.1636147,0.1636461,0.1650909,0.2963847,0.1634262)
plot(x1,y1,xlab="Times of evaluation",ylab="error of classification",type='o')