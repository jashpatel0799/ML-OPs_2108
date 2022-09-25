# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import preprocess_digits, data_viz, train_dev_test_split, h_param_tuning


# 1. set the range of hyperparameters
gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


# 2. for every combiation of hyper parameter values
hyp_para_combo = [{"gamma":g, "C":c} for g in gamma_list for c in c_list]

assert len(hyp_para_combo) == len(gamma_list)*len(c_list)



train_frac = 0.7
test_frac = 0.2
dev_frac = 0.1


# PART: -load dataset --data from files, csv, tsv, json, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)

# housekeeping
del digits



# PART: define train/dev/test splite of experiment protocol
# trin to train model
# dev to set hyperperameter of the model
# test to evaluate the performane of the model

# 80:10:10 train:dev:test


# if testing on the same as traning set: the performance matrics may overestimate 
# the goodness of the model. 
# We want to test on "unseen" sample. 

X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac, test_frac
)
  

# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()
metric = metrics.accuracy_score
best_model, best_metric, best_hyp_param, train_acc, test_acc, dev_acc = h_param_tuning(hyp_para_combo, clf, X_train, y_train, X_dev, y_dev, X_test, y_test,metric)
# if predicted < curr_predicted:
#     predicted = curr_predicted

predicted_test = best_model.predict(X_test) 

# PART: sanity check visulization of data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
# PART: Compute evaluation Matrics 
# 4. report the best set accuracy with that best model.
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted_test)}\n"
) 

print(f"Best hyperparameters were: {best_hyp_param}")


dff = {'train_acc': train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}

table = pd.DataFrame.from_dict(dff)

print(table)

print()
# min max
print(f'train  min: {min(train_acc)} \tmax:{max(train_acc)}')
print(f'test   min: {min(test_acc)} \tmax:{max(test_acc)}')
print(f'dev    min: {min(dev_acc)} \tmax:{max(dev_acc)}')

# mean
train_mean = 0
test_mean = 0
dev_mean = 0

n_train = len(train_acc)
n_test = len(test_acc)
n_dev = len(dev_acc)

train_sum = sum(train_acc)
test_sum = sum(test_acc)
dev_sum = sum(dev_acc)

train_mean = train_sum/n_train
test_mean = test_sum/n_test
dev_mean = dev_sum/n_dev

# median
median_train = 0
median_test = 0
median_dev = 0

if n_train % 2 == 0:
    median_train = train_acc[n_train//2] + train_acc[n_train//2 + 1]
else: 
    median_train = train_acc[n_train//2]

if n_test % 2 == 0:
    median_test = test_acc[n_test//2] + test_acc[n_test//2 + 1]
else: 
    median_test = test_acc[n_test//2]

if n_dev % 2 == 0:
    median_dev = dev_acc[n_dev//2] + dev_acc[n_dev//2 + 1]
else: 
    median_dev = dev_acc[n_dev//2]


print()
print(f'train mean:{median_train} \t median:{median_train}')
print(f'test  mean:{median_test}  \t median:{median_test}')
print(f'dev   mean:{median_dev}   \t median:{median_dev}')