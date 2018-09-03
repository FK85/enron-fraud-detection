# enron-fraud-detection

## Project Summary

### Goal: 
The goal of the project is to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset. This data set contains 21 financial features for 146 employees and identifies around 18 of these as person of interest (poi). We want to build a model which can predict whether a person will be a person of interest based on this data

### Why machine learning: 
Since this problem involves prediction of new data into two classes (poi and non-poi), a two-class machine learning classifier algorithm can be used to create a prediction model

### Outliers: 
The “TOTAL” record is an obvious outlier that was removed. Other than that, I removed “THE TRAVEL AGENCY IN THE PARK” as it is not an employee. In addition, I found 1 outlier where all values were either NaN or 0. (LOCKHART EUGENE E). I also removed 3 more outliers by filtering out the top 1 percentile values for total payments and total stock values.

### Missing Values: 
I found several missing values in key financial features like total payments (20), total stock value (19), salary (49) and bonus (62). Apart from that I found that 3 poi that had missing values in one of the above-mentioned fields.

## Features

### Feature Creation: 
I created two new features called “fraction from poi” and “fraction to poi”. Fraction from poi is the fraction of emails a person received from any poi out of all emails received by the person and fraction to poi is the fraction of emails a person sent to poi out of all emails sent by the person. Using these two features we can get a sense of communication frequency of a person with poi, which might be an indicator of the person being a poi

### Feature Selection: 
I used a few different methods:
1. Feature importance with decision tree and SelectKBest. I would get slightly different results for the most important features, every time I would run these. But after running it for a few times, I selected total payments, total stock value, salary, bonus, fraction of from poi and fraction to poi

SelectKBest top 10 features:
* salary : 7.27284675502
#### exercised_stock_options : 2.45275656248
#### bonus : 1.00812937457
#### total_stock_value : 2.60223651157
#### expenses : 3.32526780709
director_fees : 1.47749163845
deferred_income : 6.04514230639
long_term_incentive : 1.55849643548
fraction_from_poi : 3.78748434002
fraction_to_poi : 14.4638148346

2. Using SelectKBest in conjunction with GridSearchCV using pipeline: 7 features were selected as part of this. I used these to assess the performance. But the performance was lower as compared to my manual feature selection (I have shown the numbers at the end of this document). So, I decided to stick to my manual feature selection listed in the 1st method and commented out this block of code.
Feature Scaling: Since the financial features can have different numeric ranges, a feature with higher range can dominate the prediction model. So, I applied MinMax scaling to total payments, total stock value, salary and bonus (Fraction from/to poi was already scaled).

## Algorithm Selection: 
I ended up using Decision Tree algorithm as it was the only algorithm having a precision and recall score higher than 0.3. I also tried Gaussian Naive Bayes and Support Vector Machine.

## Tuning

### Parameter Tuning: 
Parameter tuning governs the trade off between bias and variance. Every problem set, requires its own balance between bias and variance for optimal performance. It helps in making the model fit better to the training data. If not selected intentionally, the default parameters are selected which can create a loosely fitted model leading to lower performance.

### Decision Tree Tuning: 
I tuned the minimum number of samples required to split an internal node of the tree, the minimum number of samples required to be at a leaf node and the minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. I used GridSearchCV to select the best parameter for my algorithm by providing a range of parameter values.

## Validation
I split the data into training and test data set, selecting 20% data for testing. Since the number of classes in the dataset is imbalanced with only 16 poi’s after outlier removal, I used stratified shuffle split cross-validator to make the class proportion to the whole data set balanced.

## Evaluation

### Accuracy (0.86): 
It is a percentage of correctly predicted observations out of all observations. It is only useful for a symmetric data set in terms of classes and false positives/negatives. For a data set having an imbalanced class, like our data set it is very easy to get a high accuracy without having much predictive insight. In the enron data set, most persons are non-poi. So, even if my model predicts everyone as non-poi it will have a high accuracy and still not be able to predict whether a given person is a poi or not.

### Precision (0.52): 
It is a percentage of true positive out of all predictive positive results, which indicates that if the model predicted a person as a poi, what is the probability that he is actually a poi. If my algorithm predicts a poi there is a 51% chance that he is actually a poi and a 49% chance he is not a poi. This is useful when the cost of False Positive is high. Example, if we were to use this algorithm to incriminate people in the Enron case, we want the precision to be 1. While it is ok to have a lower precision for a model stopping fraudulent credit card transactions.

### Recall (0.35): 
It is a percentage of true positive out of all actual positive results, which indicates that given a person is a poi, what is the probability of the model identifying that person as a poi. My algorithm has a 33% chance of identifying a poi and a 67% chance of not identifying a poi as a poi. This is useful when the cost of False Negative is high. Like when stopping a fraudulent transaction, you want the recall to be 1. But it is ok to have a lower recall value for the enron data set, as you know that you don’t expect the algorithm to identify every single poi.

I also tried to change the features to see the impact on performance, the performance was worse than 6 features
### 4 Features (total payments, total stock value, salary, bonus)
o Accuracy: 0.84
o Precision: 0.24
o Recall: 0.048
### 2 Features (total payments, total stock value)
o Accuracy: 0.80
o Precision: 0.09
o Recall: 0.04
### Feature Selection using SelectKBest with GridSearchCV (k = 7 was auto-selected):
o Accuracy:0.79
o Precision: 0.33
o Recall: 0.125
