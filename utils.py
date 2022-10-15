import matplotlib.pyplot as plt

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

    return X_train, y_train, X_dev, y_dev, X_dev, y_dev


def h_param_tuning(hyp_para_combo, clf, X_train, y_train, X_dev, y_dev, metric):
    best_hyp_param = None
    best_model = None
    accuracy = 0
    for curr_param in hyp_para_combo:
        # PART: setting up hyper parameter
        hyper_param = curr_param
        clf.set_params(**hyper_param)

        # PART: train model
        # 2a. train the model
        # Learn the dataset on the train subset
        clf.fit(X_train, y_train)


        # PART: get test set pridection
        # Predict the value of the digit on the test subset
        curr_predicted = clf.predict(X_dev)


        # 2b. compute accuracy on validation set
        curr_accuracy = metric(y_dev, curr_predicted)

        # 3. identify best set of hyper parameter for which validation set acuuracy is highest
        if accuracy < curr_accuracy:
            best_hyp_param = hyper_param
            accuracy = curr_accuracy
            best_model = clf
            print(f"{best_hyp_param} \tAccuracy: {accuracy}")

    return best_model, accuracy, best_hyp_param