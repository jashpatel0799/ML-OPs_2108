import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def preprocess_digits(dataset):
    # PART: data pre-processing -- to normlize data, to remove noice,
    #                               formate the data tobe consumed by model
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label


def data_viz(dataset):
    # PART: sanity check visulization of data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


def train_dev_test_split(data, label, train_frac, dev_frac, test_frac):
    dev_test_frac = 1- train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )


    fraction_want = dev_frac/(dev_frac+test_frac)
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=fraction_want, shuffle=True
    )

    return X_train, y_train, X_dev, y_dev, X_test, y_test


def h_param_tuning(hyp_para_combo, clf, X_train, y_train, X_dev, y_dev, X_test, y_test, metric):
    train_acc = []
    test_acc = []
    dev_acc = []
    best_hyp_param = None
    best_model = None
    best_train_acc, best_dev_acc, best_test_acc  = 0, 0, 0
    for curr_param in hyp_para_combo:
        # PART: setting up hyper parameter
        hyper_param = curr_param
        clf.set_params(**hyper_param)

        # PART: train model
        # 2a. train the model
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)


        # PART: get test set pridection
        # Predict the value of the digit on the test subset
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        dev_predict = clf.predict(X_dev)
        # curr_predicted = clf.predict(X_test)


        # 2b. compute accuracy on validation set
        curr_test_accuracy = metric(y_test, test_predict)
        curr_train_accuracy = metric(y_train, train_predict)
        curr_dev_accuracy = metric(y_dev, dev_predict)

        # 3. identify best set of hyper parameter for which validation set acuuracy is highest
        if best_train_acc < curr_train_accuracy:
            best_hyp_param = hyper_param
            best_train_acc = curr_train_accuracy
            best_model = clf
            print(f"{best_hyp_param} \ttrain_Accuracy: {best_train_acc}")

        if best_test_acc < curr_test_accuracy:
            best_test_acc = curr_test_accuracy
            print(f"{best_hyp_param} \ttest_Accuracy: {best_test_acc}")

        if best_dev_acc < curr_dev_accuracy:
            best_dev_acc = curr_dev_accuracy
            print(f"{best_hyp_param} \tdev_Accuracy: {best_dev_acc}")

        train_acc.append(best_train_acc)
        test_acc.append(best_test_acc)
        dev_acc.append(best_dev_acc)

    return best_model, best_train_acc, best_hyp_param, train_acc, test_acc, dev_acc