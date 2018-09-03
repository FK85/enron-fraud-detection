#!/usr/bin/python


import pickle
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import dump_classifier_and_data
from tester import test_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
enron_data.pop("TOTAL", 0) #Remove Total row
enron_data.pop("THE TRAVEL AGENCY IN THE PARK", 0) #Remove as it is not an employee name

##########################################################################

## Initial exploration

print "Total elements in the dictionary:", len(enron_data)
print "Total features in each key:", len(enron_data[enron_data.keys()[1]])
# print enron_data[enron_data.keys()[1]] #get all feature names

##########################################################################

## Create New Features

def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if poi_messages == 'NaN' or all_messages == 'NaN' or all_messages == 0:
        fraction = 0
    else:
        fraction = round(float(poi_messages) / float(all_messages), 2)

    return fraction


for name in enron_data:
    data_point = enron_data[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi
##########################################################################

## Splitting Data

features_all = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options',
            'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value',
            'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees',
            'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',
            'fraction_from_poi', 'fraction_to_poi']

features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred',
                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi',
                 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person',
                 'fraction_from_poi', 'fraction_to_poi']

data = featureFormat(enron_data, features_all)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

#############################################################################################

## Feature Selection

# Use feature importance to find the most important features
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print "Initial Decision Tree Accuracy:", accuracy_score(pred,labels_test)


importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(features_list)):
    print("%d. feature %s (%f)" % (f + 1, features_list[indices[f]], importance[indices[f]]))

# Use SelectKBest to get top 10 features

selector = SelectKBest(k=10)
selector.fit(features_train,labels_train)

scores = selector.scores_
combined = zip(features_list, scores)
scores_rank = sorted(combined, reverse=True, key=lambda x: x[1])

# for index,f in enumerate(scores_rank, start=1):
#     print index,". Feature", f[0], ":", f[1]

idxs_selected = selector.get_support(indices=True)
print "SelectKBest top 10 features:"
for x in idxs_selected:
    print features_list[x],":", scores[x]

###########################################################################################

## Outlier Removal and Missing Values

def upper_limit(array):
    """ get the 99 percentile value for a given array"""
    array = np.asarray(array, dtype='float64')
    array = array[~np.isnan(array)] # Remove NaN values
    upper_bound = np.percentile(array,99) # Set upper bound to 99 percentile value
    return upper_bound

enron_less_outliers = enron_data
enron_pre_outlier_list = []
enron_data_list = []
outlier_list = []

## Missing value variables
salary_count_nan = 0
bonus_count_nan = 0
email_count_nan = 0
tp_count_nan = 0
tsv_count_nan = 0
poi_count = 0
poi_count_nan_oth = 0
from_count_nan = 0
to_count_nan = 0

## Create a list of selected features
for name in enron_data:
    x = enron_data[name]
    enron_pre_outlier_list.append((x["poi"], x["total_payments"], x["total_stock_value"],
                            x["salary"], x["bonus"], x["fraction_from_poi"], x["fraction_to_poi"]))

## Split enron_data_list into individual feature arrays
poi1, total_payments1, total_stock_value1, salary1, bonus1, fraction_from_poi1, fraction_to_poi1 = zip(*enron_pre_outlier_list)

for name in enron_less_outliers:
    x = enron_less_outliers[name]

    outlier = True
    for value in x.values(): # Set outlier flag as True for records having all values as NaN or 0
        if value == 'NaN' or type(value) == bool or value == 0:
            None
        else:
            outlier = False

    ## If outlier, then add to the outlier_list
    if outlier or \
       (x["total_payments"] > upper_limit(total_payments1) and x["total_payments"] <> 'NaN') or \
       (x["total_stock_value"] > upper_limit(total_stock_value1) and x["total_stock_value"] <> 'NaN'):
       outlier_list.append(name)
    else:
        if x["poi"]:
            x["poi"] = 1
        else:
            x["poi"] = 0
        enron_data_list.append((x["poi"], x["total_payments"], x["total_stock_value"],
                                x["salary"], x["bonus"], x["fraction_from_poi"], x["fraction_to_poi"]))
        if x["salary"] == 'NaN':
            salary_count_nan += 1
        if x["bonus"] == 'NaN':
            bonus_count_nan += 1
        if x["email_address"] == 'NaN':
            email_count_nan += 1
        if x["total_payments"] == 'NaN':
            tp_count_nan += 1
        if x["total_stock_value"] == 'NaN':
            tsv_count_nan += 1
        if x["poi"]:
            poi_count += 1
        if x["from_messages"] == 0 or x["from_messages"] == 'NaN':
            from_count_nan += 1
        if x["to_messages"] == 0 or x["to_messages"] == 'NaN':
            to_count_nan += 1
        if x["poi"] and (x["total_payments"] == 'NaN' or x["total_stock_value"] == 'NaN' or
                                 x["from_messages"] == 'NaN' or x["salary"] == 'NaN' or x["bonus"] == 'NaN'):
            poi_count_nan_oth += 1


## Remove outliers from the dictionary
for key in outlier_list:
    enron_less_outliers.pop(key)

outlier_list.append("TOTAL")
outlier_list.append("THE TRAVEL AGENCY IN THE PARK")
print "Number of outliers: ", len(outlier_list)
print "Outliers:", outlier_list

## Count after outlier removal
print "Counts after oulier removal"
print "Total poi:", poi_count
print "Total from messages:", from_count_nan
print "Total to messages:", to_count_nan

## Print Missing Values
print "Missing Values:"
print "Missing salary_count_nan:", salary_count_nan
print "Missing bonus_count_nan:", bonus_count_nan
print "Missing email_count_nan:", email_count_nan
print "Missing total_payments_count_nan:", tp_count_nan
print "Poi wih missing values in key fields:", poi_count_nan_oth
print "total_stock_value_count_nan:", tsv_count_nan


## Split enron_data_list into individual feature arrays after outlier removal
poi,total_payments,total_stock_value,salary,bonus,fraction_from_poi,fraction_to_poi = zip(*enron_data_list)

## Convert the enron_data_list into an array and convert all values to float
enron_array = np.array(enron_data_list).astype(np.float)

###########################################################################

## Visualization

def scatter_plot(x,y,xlabel,ylabel):
    """Create a scatter plot and color data by the poi value,
        red indicates poi, blue indicates non-poi"""
    color = ['red' if i == 1 else 'blue' for i in poi]
    plt.scatter(x, y, c=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return plt.show()

scatter_plot(total_stock_value,total_payments,"total_stock_value","total_payments")
scatter_plot(salary,bonus,"salary","bonus")
scatter_plot(fraction_from_poi,fraction_to_poi,"fraction_from_poi","fraction_to_poi")

###########################################################################################

## Feature Scaling

def MinMaxScaler(X, n):
    """apply minmax scaling"""
    X = np.array(X).astype(np.float)
    X = (X - np.nanmin(X))/(np.nanmax(X) - np.nanmin(X))
    enron_array[:, n] = X
    return X

MinMaxScaler(total_payments,1)
MinMaxScaler(total_stock_value,2)
MinMaxScaler(salary,3)
MinMaxScaler(bonus,4)

###########################################################################################

## Splitting Data

enron_array = np.nan_to_num(enron_array) # Convert NaN values to zero

features = enron_array[:,1:6] # Select column 1 to 6 as features
labels = enron_array[:,0]     # Select columns 0 (poi) as label

# Apply stratified shuffle split
cv = StratifiedShuffleSplit(labels, n_iter=3, test_size=0.3, random_state=42)

for train_index,test_index in cv:
    features_train,features_test = features[train_index],features[test_index]
    labels_train,labels_test= labels[train_index],labels[test_index]


###########################################################################################

## Gaussian Naive Bayes

NB_clf = GaussianNB()
NB_clf.fit(features_train,labels_train)

NB_pred = NB_clf.predict(features_test)
print "NB Accuracy:", accuracy_score(NB_pred,labels_test)
print "NB Precision:", precision_score(labels_test, NB_pred)
print "NB Recall:", recall_score(labels_test, NB_pred)
###########################################################################################

## Support Vector Machine

#Set tuning parameters for GridSearchCV
parameters = {'kernel':['rbf', 'poly'], 'C': [100,10,1,1000], 'gamma': [0.1,1,10,100]} #,, 'linear'

svr = SVC(random_state=42, kernel='rbf')
grid_search = GridSearchCV(svr, parameters)

svm_clf = grid_search.fit(features_train, labels_train)
best_svm_clf = grid_search.best_estimator_
print "SVM Best Parameters:", grid_search.best_params_

svm_pred = best_svm_clf.predict(features_test)
print "SVM Accuracy:", accuracy_score(svm_pred,labels_test)
print "SVM Precision:", precision_score(labels_test, svm_pred)
print "SVM Recall:", recall_score(labels_test, svm_pred)

###########################################################################################
##  Decision Tree

# Set tuning parameters for GridSearchCV
parameters = {"min_samples_split": range(2,5,1),
              "min_samples_leaf": range(1,10,1),
              "min_weight_fraction_leaf": [0,0.001,0.01],#,0.001,0.01,0.04,0.1
             "criterion": ['gini']}

dt = tree.DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(dt, parameters)
dt_clf = grid.fit(features_train, labels_train)
dt_clf_best = grid.best_estimator_
print "DT Best Estimator:", dt_clf_best
print "DT Best Parameters:", grid.best_params_

## The DT algorithm results will be displayed while running tester.py
# dt_pred = dt_clf.predict(features_test)
# print "DT best accuracy:", accuracy_score(dt_pred,labels_test)
# print "DT Precision:", precision_score(labels_test, dt_pred)
# print "DT Recall:", recall_score(labels_test, dt_pred)

###########################################################################################

## Run tester.py

selected_features = ['poi', 'total_payments', 'total_stock_value',
                     'salary', 'bonus', 'fraction_from_poi', 'fraction_to_poi']
dump_classifier_and_data(dt_clf_best, enron_less_outliers, selected_features)
test_classifier(dt_clf_best, enron_less_outliers, selected_features)

##########################################################################################

###########################################################################################
# ##  Decision Tree Using SelectK in GridSearchCV
#
# print "Check performance of Decision Tree Using SelectK in GridSearchCV"
#
# data = featureFormat(enron_less_outliers, features_all, sort_keys=True)
# labels, features = targetFeatureSplit(data)
#
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
#     features, labels, test_size=0.2, random_state=42)
# dt = tree.DecisionTreeClassifier(random_state=42)
#
# pipeline = Pipeline([('select',SelectKBest()),
#                      ('dt', dt)])
#
# # Set tuning parameters for GridSearchCV
# parameters = {"select__k": range(4,10,1),
#                "dt__min_samples_split": range(2,5,1),
#                 "dt__min_samples_leaf": range(1,10,1),
#                 "dt__min_weight_fraction_leaf": [0,0.001,0.01],#,0.001,0.01,0.04,0.1
#                 "dt__criterion": ['gini']}
#
# # Use gridsearchcv to find the best parameter as well as the best value of k
# grid = GridSearchCV(pipeline, parameters, cv=cv)
# dt_clf = grid.fit(features, labels) # Fit the gridsearch with all data as there is very less data to begin with
#
# dt_step = dt_clf.best_estimator_.named_steps['select']
#
# ## Selected Features and their scores ##
# features_selected = [features_all[0]]
# for i in dt_step.get_support(indices=True):
#     features_selected.append(features_list[i])
#
# # Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
# feature_scores = ['%.2f' % elem for elem in dt_step.scores_]
# # Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
# feature_scores_pvalues = ['%.3f' % elem for elem in dt_step.pvalues_]
# # Get SelectKBest feature names, whose indices are stored in 'dt_step.get_support',
# # create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
# features_selected_tuple=[(features_list[i], feature_scores[i], feature_scores_pvalues[i]) for i in dt_step.get_support(indices=True)]
#
# # Sort the tuple by score, in reverse order
# features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]), reverse=True)
#
# # Print
# print ' '
# print 'Selected Features, Scores, P-Values'
# print features_selected_tuple
#
# #############################################################
#
# best_dt_clf = grid.best_estimator_
# print "DT Best Estimator:", best_dt_clf
# print "DT Best Parameters:", grid.best_params_
#
# # After getting the best estimator, refit the classifier to only training data using the features determined above
# data = featureFormat(enron_less_outliers, features_selected)
# labels, features = targetFeatureSplit(data)
#
# features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
#     features, labels, test_size=0.2, random_state=42)
#
# best_dt_clf = best_dt_clf.fit(features_train, labels_train)
# dt_pred = best_dt_clf.predict(features_test)
# print "DT best accuracy:", accuracy_score(dt_pred,labels_test)
# print "DT Precision:", precision_score(labels_test, dt_pred)
# print "DT Recall:", recall_score(labels_test, dt_pred)
#
# print "Because of lower performance, I am sticking to the decision tree without the selectK in pipeline"
#
# ## Run tester.py
#
# dump_classifier_and_data(best_dt_clf, enron_less_outliers, features_selected)
# test_classifier(best_dt_clf, enron_less_outliers, features_selected)
